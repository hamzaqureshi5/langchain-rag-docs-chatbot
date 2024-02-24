import os
import sys
import openai
import datetime
import warnings
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import JinaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]
JINA_EMBED_KEY = os.environ["JINA_EMBEDDING_KEY"]

sys.path.append("../..")


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

with warnings.catch_warnings(action="ignore"):
    fxn()

# fxn()


class Bot:
    chat_history = []
    answer = ""
    db_query = ""
    db_response = []

    def __init__(self):

        #        self.memory = ConversationBufferMemory()
        #        self.prompt_template = PromptTemplate()
        #        self.retrieval_qa = RetrievalQA()
        #        self.prompt = PromptTemplate()
        self.embeddings = JinaEmbeddings(
            jina_api_key=JINA_EMBED_KEY, model_name="jina-embeddings-v2-base-en"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )
        self.chroma = None
        self.llm = None
        self.QA_CHAIN_PROMPT = None
        self.vectordb = None
        #        self.chroma = Chroma.from_documents(documents=[], embedding=self.embeddings, persist_directory="chroma4/")
        #        self.chroma.persist()
        self.select_model()

    def load_PDF_doc(self, doc_path: str):
        loaders = [
            PyPDFLoader(doc_path),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )
        self.splits = self.text_splitter.split_documents(docs)
        persist_directory = "chroma4/"
        self.vectordb = Chroma.from_documents(
            documents=self.splits,
            embedding=self.embeddings,
            persist_directory=persist_directory,
        )
        self.vectordb.persist()

    def select_model(self):
        current_date = datetime.datetime.now().date()
        if current_date < datetime.date(2023, 9, 2):
            self.llm_name = "gpt-3.5-turbo-0301"
        else:
            self.llm_name = "gpt-3.5-turbo"

        # print(llm_name)

        self.llm = ChatOpenAI(model_name=self.llm_name, temperature=0)
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Must say "Thank You for using my services! Developed by Hamza Qureshi" at the end of the answer. 
        # {context}
        # Question: {question}
        # Helpful Answer:"""
        self.QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

    def retreval(self):
        self.llm = ChatOpenAI(model_name=self.llm_name, temperature=0)
        # qa_chain = RetrievalQA.from_chain_type(
        #     self.llm,
        #     retriever=self.vectordb.as_retriever(),
        #     return_source_documents=True,
        #     chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT},
        # )

        # question = "Is probability a class topic?"
        # result = qa_chain({"query": question})
        # print(result["result"])

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # ### ConversationalRetrievalChain
        from langchain.chains import ConversationalRetrievalChain

        retriever = self.vectordb.as_retriever()
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm, retriever=retriever, memory=memory
        )

    def chat(self, query: str):
        result = self.qa({"question": query, "chat_history": Bot.chat_history})
        Bot.chat_history.extend([(query, result["answer"])])
        #   db_query = result["generated_question"]
        #   db_response = result["source_documents"]
        answer = result["answer"]
        print(answer)
        return answer
