import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from research_agent import research_agent  # Make sure research_agent.py is in the same directory

st.set_page_config(page_title="🔬 Scientific Research Synthesizer", page_icon="🔬")
st.title("🔬 Scientific Research Synthesizer Agent")
st.write(
    "Ask research-heavy questions, get answers based on academic papers, web search, Wikipedia, code/calculations, and more.\n\n"
    "*Example:*\n"
    "`Summarize the latest advances in CRISPR gene editing, compare at least two newly published papers’ main findings, and check for active clinical trial studies.`"
)

# Persistent thread_id for memory & checkpointing
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.form("research_query_form"):
    user_query = st.text_area("Your research question:", height=80)
    submitted = st.form_submit_button("Ask Agent")

if submitted and user_query.strip():
    with st.spinner("Synthesizing answer..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        response = research_agent.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config
        )
        agent_messages = [m for m in response['messages'] if hasattr(m, "content")]
        final_answer = None

        if len(agent_messages) >= 2:
            # If last message is grader review ("yes"/"no") and second-to-last is the actual answer
            reviewer = agent_messages[-1].content.lower()
            if reviewer.strip().startswith("yes") or reviewer.strip().startswith("no"):
                final_answer = agent_messages[-2].content.strip()
            else:
                final_answer = agent_messages[-1].content.strip()
        elif agent_messages:
            final_answer = agent_messages[-1].content.strip()

        if final_answer:
            st.markdown(
                f"""
**Here’s a clear summary of your requests and answers:**  

{final_answer}
""",
                unsafe_allow_html=True
            )
        else:
            st.warning("Sorry, no answer could be generated. Try rephrasing your question.")

