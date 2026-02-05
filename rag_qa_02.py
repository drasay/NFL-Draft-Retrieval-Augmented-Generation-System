import chromadb
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import os
from openai import OpenAI

OPENAI_MODEL = "gpt-4"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load embedding models
model_minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model_e5 = SentenceTransformer("intfloat/e5-small-v2")

# Chroma clients
minilm_client = chromadb.PersistentClient(path="C:/python/defensive_line_prospects/chroma_minilm")
e5_client = chromadb.PersistentClient(path="C:/python/defensive_line_prospects/chroma_e5_small")

collection_minilm = minilm_client.get_collection("embeddings_minilm")
collection_e5 = e5_client.get_collection("embeddings_e5_small")

# Embed and query relevant docs
def get_context(question, embedding_model, collection, top_k=2):
    question_embedding = embedding_model.encode(question, convert_to_numpy=True)
    question_embedding = normalize([question_embedding])[0]
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    contexts = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]
    similarity_scores = [1 / (1 + d) for d in distances]
    
    annotated_contexts = [
        f"(Score: {similarity_scores[i]:.4f})\n{contexts[i]}"
        for i in range(len(contexts))
    ]
    
    context = "\n\n".join(annotated_contexts)
    return context, ids, similarity_scores

def split_context(context, max_chars=2000):
    chunks = []
    start = 0
    while start < len(context):
        end = min(start + max_chars, len(context))
        chunks.append(context[start:end])
        start = end
    return chunks


# Truncate long prompts
def truncate_prompt(prompt, tokenizer, max_tokens=1024, max_new_tokens=300):
    """
    Truncate the prompt so that prompt + generated tokens <= max_tokens.
    """
    # Calculate max allowed prompt length
    max_prompt_tokens = max_tokens - max_new_tokens
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > max_prompt_tokens:
        token_ids = token_ids[-max_prompt_tokens:]  # Keep the most recent tokens
    return tokenizer.decode(token_ids, skip_special_tokens=True)



# Prompt-style QA with Map Reduce using OpenAI
def generate_partial_answers_openai(question, context_chunks):
    partials = []
    for chunk in context_chunks:
        prompt = f"""
        You are a Super Bowl winning coach and general manager.
        DO NOT quote directly. 
        Summarize the information in your own words.
        Given the following scouting document, answer the user's query in detail.

        Document:
        {chunk}

        Question: {question}
        Answer:"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
        partials.append(response.choices[0].message.content.strip())
    return partials

def generate_final_answer_openai(question, partial_answers):
    combined = "\n\n".join(partial_answers)
    prompt = f"""
    You are an expert summarizer synthesizing multiple scouting reports.
    Based on the following individual answers, provide a clear and well-supported final answer.

    Answers:
    {combined}

    Final Answer:"""
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

# Ask question and answer from both databases
def answer_question(question):
    print(f"\nQuestion: {question}")

    # MiniLM
    context_minilm, sources_minilm, scores_minilm = get_context(question, model_minilm, collection_minilm)
    chunks_minilm = split_context(context_minilm)
    partial_answers_minilm = generate_partial_answers_openai(question, chunks_minilm)
    answer_minilm = generate_final_answer_openai(question, partial_answers_minilm)

    # E5-Small
    context_e5, sources_e5, scores_e5 = get_context(question, model_e5, collection_e5)
    chunks_e5 = split_context(context_e5)
    partial_answers_e5 = generate_partial_answers_openai(question, chunks_e5)
    answer_e5 = generate_final_answer_openai(question, partial_answers_e5)

    print("\n--- Answer from MiniLM Embeddings ---")
    print("Answer:", answer_minilm)
    print("Source Document(s) with Scores:")
    for i, (src, score) in enumerate(zip(sources_minilm, scores_minilm)):
        print(f"  {i+1}. ID: {src}, Score: {score:.4f}")

    print("\n--- Answer from E5-Small Embeddings ---")
    print("Answer:", answer_e5)
    print("Source Document(s) with Scores:")
    for i, (src, score) in enumerate(zip(sources_e5, scores_e5)):
        print(f"  {i+1}. ID: {src}, Score: {score:.4f}")

# Run CLI
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        answer_question(q)
