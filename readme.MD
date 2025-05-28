# 🤖 Agente de Pesquisa Científica com LangGraph

Este projeto foi desenvolvido como parte do **Minicurso do ENUCOMPI 2025** e demonstra como criar um agente de pesquisa científica utilizando:

- 🔍 **LangGraph** (modelo de agentes baseados em estados)
- 🧠 **ChromaDB** (banco vetorial local)
- 🤖 **Claude (Anthropic)** como modelo de linguagem
- 🌐 **Tavily AI** para busca na web
- 🧱 **LangChain** para orquestração das ferramentas
- ⏱️ **APScheduler** para agendamentos periódicos

## 📚 Objetivo

Criar um agente inteligente capaz de:

1. Realizar buscas em tempo real de artigos no **arXiv**.
2. Armazenar os artigos em um banco vetorial local (**ChromaDB**) com embeddings do HuggingFace.
3. Responder a perguntas com base nos artigos indexados.
4. Executar tarefas agendadas como reindexação ou atualizações automáticas.


## 🚀 Como executar

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/minicurso_agente.git
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

## 🧑‍🏫 Sobre o minicurso
O minicurso "Criando agentes inteligentes com LangGraph" será apresentado no ENUCOMPI 2025 e abordará desde a construção de agentes baseados em memória até sua integração com busca web e bancos vetoriais para pesquisa científica automatizada.

