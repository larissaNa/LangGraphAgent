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
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("TAVILY_API_KEY")
_set_env("ANTHROPIC_API_KEY")

@tool
def arxiv_search(query: str) -> str:
    """Busca artigos científicos no arXiv relacionados a um tópico específico, incluindo título, autores e ano."""
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3" #definir a quantidade de artigos que serão retornados pelo max_results
    response = requests.get(url)
    if response.status_code != 200:
        return "Erro ao acessar arXiv."

    root = ET.fromstring(response.content)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
    if not entries:
        return "Nenhum artigo encontrado."

    resultados = []
    for entry in entries:
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
        published = entry.find("{http://www.w3.org/2005/Atom}published").text.strip()
        year = published[:4]

        authors = entry.findall("{http://www.w3.org/2005/Atom}author")
        author_names = [author.find("{http://www.w3.org/2005/Atom}name").text.strip() for author in authors]

        resultados.append(
            f"Título: {title}\n"
            f"Autores: {', '.join(author_names)}\n"
            f"Ano: {year}\n"
            f"Link: {link}"
        )

    return "\n\n".join(resultados)

@tool
def search_ieee_articles(query: str) -> str:
    """
    Busca artigos no IEEE Xplore com base em palavras-chave.
    Retorna os títulos, autores e anos de publicação dos primeiros resultados.
    """
    api_key = "7m9hykau4u2b454wdutdfs99"
    url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

    params = {
        "apikey": api_key,
        "format": "json",
        "max_records": 5,
        "start_record": 1,
        "sort_order": "desc",
        "sort_field": "publication_year",
        "abstract": True,
        "querytext": query
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "articles" not in data:
        return "Nenhum artigo encontrado."

    results = []
    for article in data["articles"]:
        title = article.get("title", "Sem título")
        authors = ", ".join([a.get("full_name", "") for a in article.get("authors", {}).get("authors", [])])
        year = article.get("publication_year", "Ano desconhecido")
        results.append(f"- {title} ({year})\n  Autores: {authors}")

    return "\n\n".join(results)

class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- Inicializa LangGraph ---
graph_builder = StateGraph(State)

anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
tool_web = TavilySearchResults(max_results=3)
tools = [tool_web, arxiv_search, search_ieee_articles]

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", anthropic_api_key=anthropic_api_key)
llm_with_tools = llm.bind_tools(tools)

# --- Nodo de chatbot ---
def chatbot(state: State):
    new_msg = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [new_msg]}

# --- Nodo de ferramentas ---
tool_node = ToolNode(tools=tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

def route_tools(state: State):
    messages = state.get("messages", [])
    ai_message = messages[-1] if messages else None
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.set_entry_point("chatbot")

# --- Compila grafo com memória ---
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# --- Executa o agente ---
config = {"configurable": {"thread_id": "thread-001"}}
snapshot = graph.get_state(config)
print("Snapshot inicial:", snapshot)

def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        print("Assistant:", event["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Até a próxima!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print("Erro:", e)
        break