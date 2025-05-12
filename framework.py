import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command
from typing import Literal
from langgraph.graph import MessagesState, StateGraph, START, END


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

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        return "END"
    return goto

# Define functionresearch_node that takes in the current state (a dictionary with message history)
# Returns Command that either sends us to the "startup_advisor" node or the END node
def research_node(state: MessagesState) -> Command[Literal["startup_advisor", "END"]]:
    
    # Call research agent with current state and store result. The result will contain updated messages
    result = research_agent.invoke(state)
    
    # Determine which node to go to next based on the content of the last message in the result
    goto = get_next_node(result["messages"][-1], "startup_advisor")
    
    # Rename the author of the last message to "researcher" by wrapping it in a new HumanMessage
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content,  # Keep the message content the same
        name="researcher"  # Set the name to "researcher" for traceability in the conversation
    )
    
    # Return a Command object which updates the messages in state and tells the graph which node to go to
    return Command(
        update={"messages": result["messages"]},  # Update the state with the modified message list
        goto=goto,  # Set the next node to transition to
    )

# same as research_node but for the advisor agent
def advisor_node(state: MessagesState) -> Command[Literal["researcher", "END"]]:
    result = advisor_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "researcher")
    result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="startup_advisor")
    return Command(
        update ={ "messages": result["messages"]},
        goto = goto,
    )