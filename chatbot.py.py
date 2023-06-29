import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import datetime
import sqlite3
import openai
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI

# Set page title
st.title("Chatbot")

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

# Init Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY

# Initialize OpenAI embeddings and cache it as a resource
@st.cache_resource # Use st.cache_resource instead of st.cache_data
def initialize_embeddings():
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)

embeddings = initialize_embeddings()

# Set the path to the folder containing transcripts
dataPath = "C:/Users/USER/OneDrive - BIGTAPP PTE LTD/Desktop/work/work/transcript folder/"

# Cache the function that loads and splits the PDF documents into chunks and cache it as a resource
@st.cache_resource(ttl=3600) # Use st.cache_resource instead of st.cache_data and specify the ttl argument for expiration time
def load_and_split_pdfs(dataPath):
    # Initialize an empty list to store transcript pages
    pages = []

    # Iterate over all files in the folder
    for filename in os.listdir(dataPath):
        if filename.endswith(".pdf"):
            # Load and split the PDF document into chunks
            filepath = os.path.join(dataPath, filename)
            loader = PyPDFLoader(filepath)

            # Add the PDF name to the metadata of each chunk
            for chunk in loader.load_and_split():
                chunk.metadata["pdf_name"] = filename
                pages.append(chunk)
    return pages

# Call the cached function and get the pages list
pages = load_and_split_pdfs(dataPath)

# Create the embeddings using Langchain and OpenAI model and save them into a FAISS vector store and cache it as a resource
@st.cache_resource # Use st.cache_resource instead of st.cache_data
def create_faiss_vector_store(_pages, _embeddings):
    db = FAISS.from_documents(documents=pages, embedding=embeddings)
    db.save_local("faiss_index.faiss")
    return db

vectorStore = create_faiss_vector_store(pages, embeddings)

# Init openai chat model and cache it as a resource
@st.cache_resource # Use st.cache_resource instead of st.cache_data
def init_chat_model():
    return AzureChatOpenAI(
        deployment_name=OPENAI_DEPLOYMENT_NAME,
        model_name=OPENAI_MODEL_NAME,
        openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
        openai_api_version=OPENAI_DEPLOYMENT_VERSION,
        openai_api_key=OPENAI_API_KEY
    )

llm = init_chat_model()

# Load the FAISS vector store we saved into memory and cache it as a resource
@st.cache_resource # Use st.cache_resource instead of st.cache_data
def load_faiss_vector_store():
    return FAISS.load_local("faiss_index.faiss", embeddings)

vectorStore = load_faiss_vector_store()

# Use the FAISS vector store we saved to search the local document and cache it as a resource
@st.cache_resource # Use st.cache_resource instead of st.cache_data
def init_retriever(_vectorStore):
    return vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 5}, threshold=0.8, metric="cosine")

retriever = init_retriever(vectorStore)

# Use the vector store as a retriever and return the source documents and cache it as a resource
@st.cache_resource # Use st.cache_resource instead of st.cache_data
def init_qa(_llm, _retriever):
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

qa = init_qa(llm, retriever)

# Connect to a SQLite database and create a table for conversation history
conn = sqlite3.connect("conversation_history.db")
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, name TEXT, timestamp TEXT, question TEXT, answer TEXT, pdf_names TEXT)""")
conn.commit()

# Initialize the session state object with an empty list for the conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Initialize the page id attribute with a default value of None
st.session_state.page_id = None

# Create a new button in the sidebar
new_chat_button = st.sidebar.button("New Chat")

# If the button is clicked, reset the page id and the conversation history
if new_chat_button:
    st.session_state.page_id = None
    st.session_state.conversation_history = []

# Create a sidebar widget
st.sidebar.title("Conversation History")

# Query the database and get all the conversation records
c.execute("SELECT * FROM conversations")
records = c.fetchall()

# Iterate over all records in the database
for record in records:
    # Get the id, name and timestamp of the conversation
    id = record[0]
    name = record[1]
    timestamp = record[2]

    # Create a checkbox with the name and timestamp as the label
    checkbox = st.sidebar.checkbox(f"{name} ({timestamp})", value=False, key=f"{name}_{timestamp}")

    # If the checkbox is checked, show the delete and rename buttons
    if checkbox:
        # Create a delete button
        delete_button = st.sidebar.button("Delete")

        # If the delete button is clicked, delete the record from the database and refresh the page
        if delete_button:
            c.execute("DELETE FROM conversations WHERE id = ?", (id,))
            conn.commit()
            st.experimental_rerun()

        # # Create a rename button
        # rename_button = st.sidebar.button("Rename")

        # # If the rename button is clicked, get the new name from the user and update the record in the database
        # if rename_button:
        #     new_name = st.sidebar.text_input("Enter new name", value=name)
        #     confirm_button = st.sidebar.button("Confirm")
        #     # If the confirm button is clicked, update the record in the database 
        #     if confirm_button:
        #         st.experimental_rerun()
        #         c.execute("UPDATE conversations SET name = ? WHERE id = ?", (new_name, id))
        #         conn.commit()

    # Create a button with the name and timestamp as the label
    button = st.sidebar.button(f"Chat with {name}", key=f"chat_button_{id}")


    # If the button is clicked, redirect to the page with the question and answer
    if button:
        st.session_state.page_id = id

# Create a text area for the user to enter a question
question = st.text_area("Enter your question", value="", height=100)

# Create a button for the user to chat with the chatbot
if st.button("Chat"):
    if question:
        # Call the question answering model and get the result
        result = qa({"query": question})
        answer = result["result"]
        source_documents = result["source_documents"]

        # Create a set of PDF names from the source documents
        pdf_names = set()
        for doc in source_documents:
            try:
                pdf_name = doc.metadata["pdf_name"]
            except KeyError:
                pdf_name = "Unknown"
            pdf_names.add(pdf_name)
        pdf_names = sorted(list(pdf_names))

        # Get the current date and time and format it as a string
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # Add a new conversation record to the database
        c.execute("INSERT INTO conversations (name, timestamp, question, answer, pdf_names) VALUES (?, ?, ?, ?, ?)",
                  (f"Conversation {len(records) + 1}", timestamp, question, answer, ", ".join(pdf_names)))
        conn.commit()

        # Set the page id to the last inserted row id
        st.session_state.page_id = c.lastrowid
    else:
        # Display a warning message if the user does not enter a question
        st.warning("Please enter a question.")

# If the page id is set, display the corresponding question and answer from the database
if st.session_state.page_id:
    # Query the database and get the record with the page id
    c.execute("SELECT * FROM conversations WHERE id = ?", (st.session_state.page_id,))
    record = c.fetchone()

    # Get the question, answer, and pdf names from the record
    question = record[3]
    answer = record[4]
    pdf_names = record[5].split(", ")

    # Display the question and answer in the main area
    st.text_area("Question: ", value=question, height=50)
    st.text_area("Response: ", value=answer, height=200)
    # Display the PDF names in a markdown list
    st.markdown("Source files:")
    st.markdown("- " + "\n- ".join(pdf_names))

else:  # Display a message to enter a question
    st.info("Please enter a question.")