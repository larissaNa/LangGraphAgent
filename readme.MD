# 🤖 Agente de Pesquisa Científica com LangGraph

Este projeto foi desenvolvido como parte do **Minicurso do ENUCOMPI 2025** e demonstra como criar um agente de pesquisa científica utilizando:

- 🕸️ **LangGraph** (modelo de agentes baseados em estados)
- 🧠 **ChromaDB** (banco vetorial local)
- 🤖 **Claude (Anthropic)** como modelo de linguagem
- 🌐 **Tavily AI** para busca na web
- 🧱 **LangChain** para orquestração das ferramentas
- ⏱️ **APScheduler** para agendamentos periódicos

## 📚 Objetivo

Sistema Multiagente capaz de:

1. Realizar buscas em tempo real de artigos no **arXiv**.
2. Armazenar os artigos em um banco vetorial local (**ChromaDB**) com embeddings do HuggingFace.
3. Responder a perguntas gerais.
4. Executar tarefas agendadas como reindexação ou atualizações automáticas.

## 🧭 Fluxo do Sistema Multiagente

O diagrama abaixo mostra como os componentes interagem:

![Fluxograma do sistema](/images/fluxograma.png)


## 🚀 Como executar

1. Clone o repositório:

```bash
git clone https://github.com/larissaNa/LangGraphAgent.git
```
2. Crie e ative um ambiente virtual:
```bash
cd <nome_pasta>
python -m venv venv
source venv/bin/activate   # Linux/macOS
.\venv\Scripts\activate    # Windows
```
3. Instale as dependências:
```bash
pip install -r requirements.txt
```
4. Execute o agente:
```bash
python main.py
```

## 📄 Documentação do Código
#### 🔐 1. Autenticação e variáveis de ambiente
```bash
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("TAVILY_API_KEY")
_set_env("ANTHROPIC_API_KEY")
```
Função: Garante que as variáveis de API (TAVILY_API_KEY, ANTHROPIC_API_KEY) estejam definidas no ambiente, solicitando via terminal se não estiverem.

#### 🧠 2. Inicialização do modelo e ChromaDB
```bash
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="artigos_arxiv",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```
Função:
- Inicializa o modelo Claude.
- Usa sentence-transformers para vetorização.
- Cria (ou carrega) uma coleção local no ChromaDB chamada artigos_arxiv.

#### 🔧 3. Ferramentas e agentes ReAct
```bash
tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool]
agent = create_react_agent(model=llm, tools=tools)

```
Função: Registra a ferramenta TavilySearch como agente ReAct capaz de responder a buscas web.

#### 📚 4. Busca e ingestão no arXiv
```bash
class ArxivIngestInput(BaseModel):
    query: str
    max_results: int = 3

@tool("arxiv_search_and_ingest", args_schema=ArxivIngestInput)
def arxiv_search_and_ingest(query: str, max_results: int = 3) -> str:
    ...
```
Função:

- Realiza uma busca na API do arXiv com o termo informado.
- Extrai título, autores, resumo e link de cada artigo.
- Armazena resumos vetorizados no ChromaDB.
- Evita duplicatas via adicionados_arxiv_links.

#### ⏱️ 5. Agendamento com APScheduler
```bash
scheduler = BackgroundScheduler()
scheduler.start()
active_jobs = {}
```
Função: Inicia um agendador em segundo plano para execuções periódicas.

#### 📥 Agendar pesquisa recorrente
```bash

@tool
def schedule_research(mensagem: str) -> str:
    ...
```
Função:

- Reconhece comandos no formato:
    - "pesquise sobre [tema] durante [N] minutos a cada [M] segundos"
    - Executa arxiv_search_and_ingest() a cada intervalo, por tempo limitado.

#### ❌ Cancelar pesquisa agendada
```bash
@tool
def cancel_research(mensagem: str) -> str:
    ...

```
Função: Cancela tarefas agendadas ativamente com base no nome do tema.

#### 🤖 6. Criação dos Agentes
```bash
tavily_agent = create_react_agent(
    model=llm,
    tools=[tavily_tool],
    prompt= 'You perform web searches',
    name="tavily_agent"
)

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

```
Função: Cria três agentes:
-  Tavily: busca na web.
- arXiv: ingestão de artigos.
- Scheduler: agenda e cancela pesquisas.

#### 🧑‍⚖️ 7. Supervisor
```bash
supervisor_graph = create_supervisor(
    model=llm,
    agents=[tavily_agent, arxiv_agent, sched_agent],
    prompt="Você é um supervisor que delega...",
    output_mode="full_history"
)
compiled_supervisor = supervisor_graph.compile()
```
Função: Controla qual agente executa a tarefa, com base no conteúdo da mensagem. Gera histórico completo da conversa.

#### 🔄 8. Grafo com LangGraph
```bash
class StateSchema(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph = StateGraph(StateSchema)
graph.add_node("supervisor", compiled_supervisor)
graph.add_edge(START, "supervisor")
compiled = graph.compile(checkpointer=MemorySaver())
```
Função:

- Define o estado inicial com histórico de mensagens.
- Cria um grafo com o supervisor como nó principal.
- Usa MemorySaver para manter estado por sessão (thread).

#### 🧪 9. Loop interativo
```bash
def run(user_input: str):
    config = {"configurable": {"thread_id": "scheduler-thread"}}
    for chunk in compiled.stream({"messages": [...]}, config=config):
        ...

if __name__ == "__main__":
    ...
```
Função:

- Permite interação pelo terminal.
- Lê comandos do usuário e executa o fluxo definido no grafo.
- Suporta múltiplas interações e threads com memória.

## 🧾 Exemplos de comandos suportados pelo usuário
```bash
"pesquise sobre redes neurais durante 2 minutos a cada 30 segundos"
```
```bash
"cancelar busca sobre redes neurais"
```
```bash
"busque papers sobre transformers"
```
```bash
"Como esta o clima hoje em Piripiri Piauí?"
```

## 🧑‍🏫 Sobre o minicurso
O minicurso "Criando agentes inteligentes com LangGraph" será apresentado no ENUCOMPI 2025 e abordará desde a construção de agentes baseados em memória até sua integração com busca web e bancos vetoriais para pesquisa científica automatizada.

