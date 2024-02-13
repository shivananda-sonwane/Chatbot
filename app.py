from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone 
from langchain.vectorstores import Pinecone
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain import hub



load_dotenv()


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


def doc_preprocessing():
    """
    The function `doc_preprocessing` loads PDF documents from a specified directory, splits the text
    into chunks of 1000 characters with no overlap, and returns the split documents.
    :return: a list of documents that have been preprocessed.
    """
    loader = DirectoryLoader(
        'data/',
        glob='data\MySQLNotesForProfessionals.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embedding_db():
    """
    The function `embedding_db` initializes and returns a Pinecone document database using the OpenAI
    embedding model.
    :return: The function `embedding_db()` returns the document database object `doc_db`.
    """
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(
        docs_split, 
        embeddings, 
# enter your index_name here
        index_name='chatbot'
        
    )
    return doc_db

llm = ChatOpenAI()
doc_db = embedding_db()
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

def retrieval_answer(query):
    """
    The function `retrieval_answer` takes a query as input and uses a retrieval-based question answering
    model to find the answer to the query.
    
    :param query: The query parameter is the question or query that you want to ask the retrieval-based
    question answering model
    :return: The result of the retrieval_answer function is being returned.
    """
    qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=doc_db.as_retriever(search_type="similarity", search_kwargs={"k":3}),
    chain_type_kwargs={"prompt": prompt},
    )
    query = query
    result = qa.run(query)
    return result

def main():
    """
    The main function creates a question and answering app powered by LLM and Pinecone, where users can
    input a query and receive an answer.
    """
    st.title("Question and Answering App")

    text_input = st.text_input("Ask your query...") 
    if st.button("Ask Query"):
        if len(text_input)>0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)

if __name__ == "__main__":
    main()

    







