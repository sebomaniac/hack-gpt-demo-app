import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import create_react_agent


# create title and description
st.title("Startup Evaluator")
st.write("Enter your startup/business idea below to view insights.")

# create text input to enter idea
user_prompt = st.text_input("Enter your business or startup idea:", "")
submit = st.button("Submit", disabled=(not user_prompt))

# create sidebar for API keys
st.sidebar.header("API Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
tavily_api_key = st.sidebar.text_input("Tavily API Key", type="password")

# takes the keys the user typed in and sets them as environment variables
# necessary when using langchain and tavily because they read keys from the environment variables
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key

# check if API keys are set, if not, show warning message and stop execution
if not openai_api_key or not tavily_api_key:
    st.warning("Please enter your API keys in the sidebar to continue.")
    st.stop()

# initialize models
llm = ChatOpenAI(model="gpt-4o")
tavily_tool = TavilySearchResults(max_results=5)

# create common system prompt for both agents
def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants. "
        "Use the provided tools to progress towards answering the question. "
        "If you are unable to fully answer, another assistant with different tools will help. "
        "If the final answer is found, prefix your response with FINAL ANSWER."
        f"\n{suffix}"
    )

# create agents
research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=make_system_prompt(
        "You are a research assistant helping evaluate startup ideas. "
        "Your job is to gather market data, trends, and competitors. Do NOT make recommendations."
    ),
)

advisor_agent = create_react_agent(
    llm,
    tools=[],
    prompt=make_system_prompt(
        "You are a startup advisor evaluating business ideas. "
        "Make a decision based on the research: pursue or not pursue. Be objective and logical."
    ),
)