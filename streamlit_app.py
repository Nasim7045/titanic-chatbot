import os
import pandas as pd
import streamlit as st
from fastapi import FastAPI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

# ✅ 1️⃣ Load Titanic Dataset
DATA_PATH = "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

if not os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

# ✅ 2️⃣ Set API Key Correctly
GEMINI_API_KEY = "api-key"  # Directly use the key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  # Set environment variable

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

# ✅ 3️⃣ Create LangChain Pandas Agent
agent = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True,
    handle_parsing_errors=True,
    allow_dangerous_code=True
)

# ✅ 4️⃣ FastAPI Backend
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query/")
async def query_titanic(request: QueryRequest):
    try:
        response = agent.run(request.question)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}

# ✅ 5️⃣ Streamlit UI
st.title("🚢 Titanic Data Chatbot")
st.write("Ask any question about the Titanic dataset!")

question = st.text_input("Enter your question:")
if st.button("Ask"):
    if question:
        try:
            response = agent.run(question)
            st.write(response)
        except Exception as e:
            st.write(f"Error: {str(e)}")
