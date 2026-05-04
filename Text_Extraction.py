from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader(r"C:\Users\Saatvik\OneDrive\Desktop\Saatvik_CV.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings( model_name="all-MiniLM-L6-v2")
n=len(chunks)
vectors=[]
for i in range(0,n):
    chunk=chunks[i].page_content
    embedding=embeddings.embed_query(chunk)
    vectors.append(embedding)

vector_store = FAISS.from_documents(chunks, embeddings)
query="What are projects worked upon"
results=vector_store.similarity_search(query,k=3)

retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 3})

query="What are the qualifications of the person"
results=retriever.invoke(query)
for i in results:
    print(i.page_content)