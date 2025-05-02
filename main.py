#langgraph, Tavily AI, anthropic, crhomaDB
#Busca via Tavily (web search)
#Gerenciamento de estado com memória por thread (MemorySaver)

import getpass
import os
import requests
import xml.etree.ElementTree as ET
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("TAVILY_API_KEY")
_set_env("ANTHROPIC_API_KEY")

# --- Inicializa embeddings e ChromaDB ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="artigos_arxiv",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# --- Ferramenta: busca na web com Tavily ---
tool_web = TavilySearchResults(max_results=3)

@tool
def arxiv_search_and_ingest(query: str, max_results: int = 3) -> str:
    """
    Busca artigos no arXiv e armazena suas informações no banco vetorial ChromaDB.
    """
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    if response.status_code != 200:
        return "Erro ao acessar arXiv."

    root = ET.fromstring(response.content)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
    if not entries:
        return "Nenhum artigo encontrado no arXiv."

    documents = []
    for entry in entries:
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
        published = entry.find("{http://www.w3.org/2005/Atom}published").text.strip()
        year = published[:4]
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()

        authors = entry.findall("{http://www.w3.org/2005/Atom}author")
        author_names = [author.find("{http://www.w3.org/2005/Atom}name").text.strip() for author in authors]
        authors_str = ", ".join(author_names)

        keywords = [cat.attrib.get("term") for cat in entry.findall("{http://www.w3.org/2005/Atom}category")]
        metadata = {
            "title": title,
            "link": link,
            "year": year,
            "authors": authors_str,
            "abstract": summary,
            "keywords": ", ".join(filter(None, keywords)) or "N/A"
        }

        documents.append(Document(page_content=summary, metadata=metadata))

    # Split e salva os documentos
    text_splitter = RecursiveCharacterTextSplitter()
    texts = text_splitter.split_documents(documents)

    vectorstore.add_documents(texts)
    vectorstore.persist()

    return f"{len(documents)} artigos sobre '{query}' foram adicionados ao banco vetorial com sucesso."

@tool
def semantic_search(query: str, k: int = 5) -> str:
    """Realiza busca semântica nos documentos do ChromaDB."""
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "Nenhum documento encontrado."
    output = []
    for doc in docs:
        meta = doc.metadata
        output.append(
            f"- {meta.get('title')} ({meta.get('year')})\n  Autores: {meta.get('authors')}\n  Link: {meta.get('link')}\n  Trecho: {doc.page_content[:200]}..."
        )
    return "\n\n".join(output)

class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- Inicializa LangGraph ---
graph_builder = StateGraph(State)

# Inicializa LLM Anthropic
anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", anthropic_api_key=anthropic_api_key)

# Adiciona ferramentas ao LLM
tools = [tool_web, arxiv_search_and_ingest, semantic_search]
llm_with_tools = llm.bind_tools(tools)

# --- Nodo de chatbot ---
def chatbot(state: State):
    new_msg = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [new_msg]}

# Nodo de ferramentas
tool_node = ToolNode(tools=tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

def route_tools(state: State):
    msgs = state.get("messages", [])
    ai_msg = msgs[-1] if msgs else None
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.set_entry_point("chatbot")

# Compila grafo com memória
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Configuração e execução interativa
config = {"configurable": {"thread_id": "thread-001"}}
print("Snapshot inicial:", graph.get_state(config))

def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for ev in events:
        print("Assistant:", ev["messages"][-1].content)

if __name__ == "__main__":
    while True:
        try:
            ui = input("User: ")
            if ui.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(ui)
        except Exception as e:
            print("Erro:", e)
            break