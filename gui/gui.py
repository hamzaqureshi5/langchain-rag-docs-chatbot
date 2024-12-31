
import os

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.chains import retrieval_qa
# Run chain
from langchain.chains import RetrievalQA


import os
import io
import chainlit as cl
import PyPDF2
from io import BytesIO


from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

LANG_CHAIN_API_KEY = os.getenv('LANG_CHAIN_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
JINA_EMBEDDING_KEY = os.getenv('JINA_EMBEDDING_KEY')


from langchain.embeddings import JinaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

embeddings = JinaEmbeddings(jina_api_key = JINA_EMBEDDING_KEY, model_name="jina-embeddings-v2-base-en")



llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# embeddings = OpenAIEmbeddings()

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

def process_file(file):
    # Log the available attributes for debugging
    print(dir(file))
    try:
        # Attempt to access the content attribute
        content = file.content
    except AttributeError:
        # If content is unavailable, fallback to read or another method
        content = file.read() if hasattr(file, "read") else None
    
    if content is None:
        raise ValueError("Unable to extract file content.")

    # Write content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tempfile:
        tempfile.write(content)

    return tempfile.name

# def process_file(file: AskFileResponse):
#     import tempfile

#     if file.type == "text/plain":
#         Loader = TextLoader
#     elif file.type == "application/pdf":
#         Loader = PyPDFLoader

#     with tempfile.NamedTemporaryFile() as tempfile:
#         tempfile.write(file.content)
#         loader = Loader(tempfile.name)
#         documents = loader.load()
#         docs = text_splitter.split_documents(documents)
#         for i, doc in enumerate(docs):
#             doc.metadata["source"] = f"source_{i}"
#         return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file

    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch



@cl.on_chat_start
async def start():
    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path="./r.jpg")
    ]
    await cl.Message(content="Hello there, Welcome to AskAnyQuery related to Data!", elements=elements).send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)

    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=docsearch.as_retriever(),
        #return_source_documents=True,
        #chain_type="stuff",

    )

    # chain = RetrievalQAWithSourcesChain.from_chain_type(
    #     llm,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(max_tokens_limit=4097),
    # )

    # Let the user know that the system is ready
    
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the documents from the user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()