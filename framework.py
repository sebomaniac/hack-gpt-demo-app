import os
import streamlit as st

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
# necessary using langchain and tavily because they read keys from the environment variables
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key

# check if API keys are set, if not, show warning message and stop execution
if not openai_api_key or not tavily_api_key:
    st.warning("Please enter your API keys in the sidebar to continue.")
    st.stop()