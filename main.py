import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import pinecone
import openai

model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-west1-gcp")
index_name = "website-content"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)  
index = pinecone.Index(index_name)

openai.api_key = "YOUR_OPENAI_API_KEY"

def crawl_and_embed(url):
    print(f"Crawling {url}...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.get_text()
    chunks = [content[i:i + 500] for i in range(0, len(content), 500)]

    print(f"Embedding and storing chunks from {url}...")
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        metadata = {"url": url, "content": chunk}
        index.upsert([(f"{url}-chunk-{i}", embedding, metadata)])

def handle_query(query):
    print(f"Query: {query}")

    query_embedding = model.encode(query).tolist()

    results = index.query(query_embedding, top_k=5, include_metadata=True)
    retrieved_content = [match['metadata']['content'] for match in results['matches']]

    context = "\n\n".join(retrieved_content)
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer the question using the context provided." 

    response = openai.Completion.create(
        engine="text-davinci-003",  
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    answer = response.choices[0].text.strip()
    return answer

def main():
    
    websites = [
        "https://www.uchicago.edu/",
        "https://www.washington.edu/",
        "https://www.stanford.edu/",
        "https://und.edu/"
    ]
    for website in websites:
        crawl_and_embed(website)

    user_query = input("Enter your query: ")
    response = handle_query(user_query)
    print("\nResponse:\n", response)

if __name__ == "__main__":
    main()
