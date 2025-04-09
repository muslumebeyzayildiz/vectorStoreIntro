from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma #vector database i
from langchain_openai import OpenAIEmbeddings

load_dotenv()


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},#verinin ekstra bilgisi gibi
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
    ),#bunları vektörize ettikten sonra vektörlerin saklandıkları herhangi bir aracın/database in içerisine EMMBED ETMEK GÖMMEK İSTİYORUZ
    #bu dökümanları alıp bir VECTOR STORES un içine koyacağız

    #chroma  güzel bir veri tabanı...
]


vectorstore = Chroma.from_documents(
    documents=documents,#dökümanı aldık
    embedding=OpenAIEmbeddings(),#vektörler çevirdik
)#croma db ye yükle dedik

if __name__ == "__main__":
    #print(vectorstore.similarity_search("cat"))#benzerlik araması
    #print(vectorstore.similarity_search_with_score("cat"))
    embedding = OpenAIEmbeddings().embed_query("cat")
    print(vectorstore.similarity_search_by_vector(embedding))

    #RAG -> retrieval augmented generation
    #RAG -> i daha iyi anlamak için bunu yaptık.
    #kendi firmma için bir chatbor yapmak istiyorum.
    #bir vector store a ekstra bilgi veririm. ve LLM e Ofis ile ilgili bir soru geldiğinde eğer bilemiyorsan. al işte buradan bak (benim sana verdiğim document kısmından)
    #işte buna retrival yapmak diyoruz. llm kendi eğitimde bulunmayan biligiyi de harmanlayarak
    # .
