import os
import openai
import sys
sys.path.append('../..')

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

with warnings.catch_warnings(action="ignore"):
    fxn()

fxn()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
KEY  = os.environ['JINA_EMBEDDING_KEY']

from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("documents/temp.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


# Split this Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)
# print(len(splits))

from langchain.embeddings import JinaEmbeddings
embedding = JinaEmbeddings(jina_api_key = KEY, model_name="jina-embeddings-v2-base-en")

from langchain.vectorstores import Chroma
# persist_directory = 'docs/chroma1/'
persist_directory = 'chroma4/'
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
len(docs)

vectordb.persist()

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)

from langchain.chains import RetrievalQA

# Build prompt
from langchain.prompts import PromptTemplate

# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "Fuck you !" at the end of the answer. 
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Must say "Fuck You!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(    input_variables=["context", "question"],    template=template,)


# Run chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# question = "Is probability a class topic?"
# result = qa_chain({"query": question})
# print(result["result"])


# ### Memory
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# ### ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain
retriever = vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)


chat_history = []
answer = ""
db_query = ""
db_response = []


query ="what is this document?"

def chat(query:str):
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.extend([(query, result["answer"])])
#   db_query = result["generated_question"]
#   db_response = result["source_documents"]
    answer = result["answer"]
    print(answer)
    return answer

chat(query)
