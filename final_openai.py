from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import find_dotenv, load_dotenv

from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


load_dotenv(find_dotenv())

INDEX_FOLDER = "final_finhay"


def setup():
    loader = TextLoader("./examples/finhay_en.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separator="\n\n"
    )
    chunks = text_splitter.split_documents(documents)
    # Get embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPEN_API_KEY"))

    # Check for INDEX_FOLDER and retrieve db
    if not os.path.exists(INDEX_FOLDER) or not os.path.isdir(INDEX_FOLDER):
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(INDEX_FOLDER)
    else:
        db = FAISS.load_local(INDEX_FOLDER, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPEN_API_KEY"),
        temperature=0,
        max_tokens=400,
        model="gpt-3.5-turbo",
    )
    streaming_llm = OpenAI(
        openai_api_key=os.environ.get("OPEN_API_KEY"),
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0,
    )
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer (in Vietnamese):"""
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

    qa = ConversationalRetrievalChain(
        retriever=db.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        memory=memory
    )
    return qa


def ask_finhay(qa: BaseRetrievalQA, text: str, chat_history) -> str:
    result = qa({"question": text, "chat_history": chat_history})
    chat_history.append((text, result["answer"]))
    return result["answer"]


def main():
    st.set_page_config(page_title="QnA Finhay", page_icon="💰")
    st.header("QnA Finhay")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "histories" not in st.session_state:
        st.session_state.histories = []

    if "qa" not in st.session_state:
        st.session_state.qa = setup()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Wait for it..."):
            response = ask_finhay(
                st.session_state.qa, prompt, st.session_state.histories
            )
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
