from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI #gerçek bir örnek için. Gerçekten LLM e yollayacağız.
from langchain_core.runnables import RunnableLambda #
from langchain_core.prompts import ChatPromptTemplate #
from langchain_core.runnables import RunnablePassthrough #

load_dotenv()


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

#koyduğmuz dökümanları (vector store dan okuyabilecek) alabilen bir obje retriever
retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result
#RunnableLambda ın ne yapmasını istediğini neyi çalıştırması istediğini koyuyuoruz
#hep ilk sonucu almak için ".bind" argümanıını veririp (k=1) ile en top ı getir

#print(retriever.batch(["cat", "shark"]))
#batch ile birden fazla arama yapbiliyorum


llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
#bu soruyu sadece sana verilen contexde cevapla(benim verdiğim örnekleri)
message = """
Answer this question using the provided context only.

{question}

Context: 

{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
#prompt a context ve questions vermeliyim
if __name__ == "__main__":
    response = rag_chain.invoke("tell me about cats")
    print(response.content)