#  Scientific Research Synthesizer Agent

A powerful, memory-enabled conversational AI assistant that can answer research queries with up-to-date, multi-source evidence—including academic papers (arXiv, Semantic Scholar, PubMed), Wikipedia, real-time web search with Tavily, and on-the-fly Python calculations. Built using [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and [Streamlit](https://streamlit.io/).

---

##  Features

- **ReAct-Style Multi-Step Reasoning**: The agent reasons, selects tools, and synthesizes multi-source scientific answers, step by step.
- **Advanced Tools**:
  - Arxiv, Semantic Scholar, and PubMed paper search
  - Wikipedia search and summary
  - Tavily real-time web search
  - Python REPL for calculations and data crunching
- **Self-Reflective Grader**: The agent grades its own answers and loops until it provides a sufficiently complete response.
- **Memory Enabled**: Each session gets a unique thread ID—persistence and context for multi-turn or iterative research.
- **One-Click Web UI**: Clean Streamlit app with only the agent's final, best answer shown for each question.

---

##  Installation
    ```bash

1. **Clone this repo**
    ```bash
    git clone https://github.com/Viswa-Prakash/scientific_research_synthesizer_agent.git
    cd scientific_research_synthesizer_agent
    

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt


3. **Configure API Keys**  
   Create a `.env` file in the root folder:
    ```
    OPENAI_API_KEY=sk-xxxx
    TAVILY_API_KEY=your-tavily-key
    # Any other keys you may use for additional services
    ```

---
##  Usage

#### **1. Start the Streamlit app**
```bash
streamlit run app.py

#### **2. Ask a multi-line research question**

Summarize the most recent breakthroughs in quantum computing from academic literature.
Also, how does quantum computing differ from classical computing?
Find a relevant Wikipedia summary, and show one real-world business application.

#### **3. Get a single, clear, final answer**
The app displays only the last agent message—a comprehensive, step-by-step synthesis using all the tools available.
---