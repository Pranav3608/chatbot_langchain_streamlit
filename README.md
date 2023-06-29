# Chatbot using Langchain, Azure OpenAI and Streamlit

This project is a chatbot that can answer questions based on a set of PDF documents. It uses Langchain to load and split the PDF documents into chunks, create embeddings using Azure OpenAI model, and store them in a FAISS vector store. It also uses Azure OpenAI to create a question answering model that can retrieve the relevant chunks and generate answers. The chatbot interface is built using Streamlit and allows the user to enter questions, view answers, and manage the conversation history.

## Installation

To run this project, you need to have Python 3.7 or higher and pip installed. You also need to have an Azure account and an OpenAI API key. You can follow these steps to install the required dependencies and set up the environment variables:

1. Clone this repository or download the zip file.
2. Navigate to the project folder and create a virtual environment using `python -m venv env`.
3. Activate the virtual environment using `env\Scripts\activate` on Windows or `source env/bin/activate` on Linux/MacOS.
4. Install the required packages using `pip install -r requirements.txt`.
5. Create a `.env` file in the project folder and add the following variables:

```
OPENAI_API_KEY=<your OpenAI API key>
OPENAI_DEPLOYMENT_ENDPOINT=<your OpenAI deployment endpoint>
OPENAI_DEPLOYMENT_NAME=<your OpenAI deployment name>
OPENAI_MODEL_NAME=<your OpenAI model name>
OPENAI_EMBEDDING_DEPLOYMENT_NAME=<your OpenAI embedding deployment name>
OPENAI_EMBEDDING_MODEL_NAME=<your OpenAI embedding model name>
OPENAI_DEPLOYMENT_VERSION=<your OpenAI deployment version>
```

6. Save the `.env` file and close it.

## Usage

To run the chatbot, you need to have a folder with some PDF documents that you want the chatbot to use as the knowledge base. You can use any PDF documents that are relevant to your domain or topic of interest. You also need to specify the path to the folder in the `dataPath` variable in the code.

Once you have prepared the PDF documents and set the path, you can run the chatbot using `streamlit run chatbot.py`. This will launch a web browser and display the chatbot interface. You can enter your questions in the text area and click on the "Chat" button to get an answer from the chatbot. You can also view and manage your conversation history in the sidebar. You can create a new chat, delete or rename an existing chat, or select a chat to view its question and answer.
