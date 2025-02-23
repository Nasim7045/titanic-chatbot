import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

# Load Titanic dataset
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# Function to generate base64 encoded plot
def generate_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

# Function to process user queries
def analyze_query(question):
    question = question.lower()

    if "percentage of passengers were male" in question:
        male_percentage = (df["Sex"] == "male").mean() * 100
        return {"text": f"{male_percentage:.2f}% of passengers were male."}

    elif "histogram of passenger ages" in question:
        fig, ax = plt.subplots()
        sns.histplot(df["Age"].dropna(), bins=20, kde=True, ax=ax)
        ax.set_title("Histogram of Passenger Ages")
        return {"image": generate_plot(fig)}

    elif "average ticket fare" in question:
        avg_fare = df["Fare"].mean()
        return {"text": f"The average ticket fare was ${avg_fare:.2f}."}

    elif "passengers embarked from each port" in question:
        fig, ax = plt.subplots()
        sns.countplot(x=df["Embarked"].dropna(), ax=ax)
        ax.set_title("Passengers per Embarkation Port")
        return {"image": generate_plot(fig)}

    else:
        return {"text": "I'm not sure how to answer that. Try asking about fares, ages, or embarkation points."}

# Streamlit UI
st.title("🚢 Titanic Chatbot")
st.write("Ask me questions about the Titanic dataset!")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question:
        response = analyze_query(question)

        if "text" in response:
            st.write(response["text"])
        
        if "image" in response:
            image_bytes = base64.b64decode(response["image"])
            st.image(io.BytesIO(image_bytes), caption="Visualization", use_column_width=True)
