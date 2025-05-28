import os
import getpass
import dotenv
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import arxiv
from pydantic import BaseModel, Field
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# --- Autenticação ---
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("TAVILY_API_KEY")
_set_env("ANTHROPIC_API_KEY")

# Mantém links já adicionados
adicionados_arxiv_links = set()

# --- Modelo base ---
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# 3. Configura ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="artigos_arxiv",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# --- Ferramentas ---
tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool]

# --- Agente ReAct ---
agent = create_react_agent(model=llm, tools=tools)

# 4.3 Busca e ingestão no arXiv + ChromaDB
class ArxivIngestInput(BaseModel):
    query: str = Field(..., description="Termo de pesquisa para artigos no arXiv")
    max_results: int = Field(3, description="Número máximo de artigos a buscar")

@tool("arxiv_search_and_ingest", args_schema=ArxivIngestInput)
def arxiv_search_and_ingest(query: str, max_results: int = 3) -> str:
    """
    Busca artigos no arXiv e os armazena no ChromaDB.
    Retorna título, autores, ano, abstract e link de cada novo artigo.
    """
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{query}&max_results={max_results}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return f"Erro ao acessar arXiv: {r.status_code}"

    root = ET.fromstring(r.content)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
    if not entries:
        return "Nenhum artigo encontrado no arXiv."

    new_docs = []
    outputs = []
    for e in entries:
        link = e.find("{http://www.w3.org/2005/Atom}id").text.strip()
        if link in adicionados_arxiv_links:
            continue

        title = e.find("{http://www.w3.org/2005/Atom}title").text.strip()
        published = e.find("{http://www.w3.org/2005/Atom}published").text.strip()
        year = published[:4]
        summary = e.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        authors = [a.find("{http://www.w3.org/2005/Atom}name").text.strip()
                   for a in e.findall("{http://www.w3.org/2005/Atom}author")]

        # preenche vectorstore
        meta = {
            "title": title,
            "authors": ", ".join(authors),
            "year": year,
            "link": link
        }
        new_docs.append(Document(page_content=summary, metadata=meta))
        adicionados_arxiv_links.add(link)

        outputs.append(
            f"- {title} ({year})\n"
            f"  Autores: {', '.join(authors)}\n"
            f"  Abstract: {summary[:300]}...\n"
            f"  Link: {link}"
        )

    if not new_docs:
        return "Nenhum artigo novo foi adicionado (já existente)."

    # Ingestão em Chroma
    splitter = RecursiveCharacterTextSplitter()
    docs = splitter.split_documents(new_docs)
    vectorstore.add_documents(docs)
    vectorstore.persist()

    return "Novos artigos adicionados:\n\n" + "\n\n".join(outputs)

# ---------- 5. Ferramentas de agendamento ----------
scheduler = BackgroundScheduler()
scheduler.start()
active_jobs = {}

@tool
def schedule_research(mensagem: str) -> str:
    """
    Comando: pesquise sobre X durante N minutos a cada M segundos
    """
    padrao = re.compile(
        r"pesquise sobre (.*?) durante (\d+) minutos?.*?a cada (\d+) segundos?",
        re.IGNORECASE
    )
    m = padrao.search(mensagem)
    if not m:
        return "Use: 'pesquise sobre [tema] durante [N] minutos a cada [M] segundos'."
    tema, dur_min, int_seg = m.groups()
    dur_min, int_seg = int(dur_min), int(int_seg)
    job_id = f"job_{tema.replace(' ','_')}_{uuid.uuid4().hex[:6]}"
    fim = datetime.now() + timedelta(minutes=dur_min)
    start_idx = {"value": 0}

    def tarefa():
        if datetime.now() >= fim:
            scheduler.remove_job(job_id)
            active_jobs.pop(tema, None)
            print(f"🛑 Tarefa '{tema}' finalizada.")
            return
        print(f"[Scheduler] buscando '{tema}' (start={start_idx['value']})")
        resultado = arxiv_search_and_ingest.func(tema, 3)
        start_idx["value"] += 3
        print(resultado)

    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
    scheduler.add_job(tarefa, 'interval', seconds=int_seg, id=job_id)
    active_jobs[tema] = job_id
    return f"✅ Agendada: '{tema}' por {dur_min}min a cada {int_seg}s."

@tool
def cancel_research(mensagem: str) -> str:
    """
    Comando: cancelar busca sobre X
    """
    m = re.search(r"cancelar busca sobre (.+)", mensagem, re.IGNORECASE)
    if not m:
        return "Use: 'cancelar busca sobre [tema]'."
    tema = m.group(1).strip()
    jid = active_jobs.get(tema)
    if not jid:
        return f"Nenhuma tarefa ativa para '{tema}'."
    scheduler.remove_job(jid)
    del active_jobs[tema]
    return f"❌ Tarefa para '{tema}' cancelada."

# Agente Tavily (busca geral)
tavily_agent = create_react_agent(
    model=llm,
    tools=[tavily_tool],
    prompt= 'You perform web searches',
    name="tavily_agent"
)

# Agente arXiv (pesquisa científica)
arxiv_agent = create_react_agent(
    model=llm, tools=[arxiv_search_and_ingest], name="arxiv_agent",
    prompt="You search, ingest into ChromaDB and report arXiv papers."
)

sched_agent  = create_react_agent(
    llm,
    tools=[schedule_research, cancel_research],
    prompt="You schedule or cancel periodic arXiv searches.",
    name="scheduler_agent"
)

# Supervisor
# --- 5. Cria Supervisor ---
supervisor_graph = create_supervisor(
    model=llm,
    agents=[tavily_agent, arxiv_agent, sched_agent],
    prompt=(
        "Você é um supervisor que delega ao agente mais adequado entre:\n"
        "- Tavily (buscas gerais)\n"
        "- arXiv (buscas imediatas)\n"
        "- Scheduler (buscas agendadas)\n"
        "Após obter a resposta, sempre gere UMA mensagem final clara ao usuário."
    ),
    add_handoff_messages=True,
    add_handoff_back_messages=True,
    output_mode="full_history"
)

# Compilar o gráfico do supervisor para torná-lo executável
compiled_supervisor = supervisor_graph.compile()

# --- 6. Monta StateGraph e compila ---
class StateSchema(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph = StateGraph(StateSchema)
graph.add_node("supervisor", compiled_supervisor)
graph.add_edge(START, "supervisor")
compiled = graph.compile(checkpointer=MemorySaver())

# ---------- 9. Loop interativo ----------
def run(user_input: str):
    config = {"configurable": {"thread_id": "scheduler-thread"}}
    try:
        for chunk in compiled.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="values"
        ):
            for msg in chunk.get("messages", []):
                msg.pretty_print()
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")

if __name__ == "__main__":
    print("⚙️ Multi-Agente com Scheduler ativo! Digite 'exit' para sair.\n")
    while True:
        ui = input("User: ")
        if ui.lower() in ["exit","quit","q"]:
            print("Encerrando...")
            break
        run(ui)