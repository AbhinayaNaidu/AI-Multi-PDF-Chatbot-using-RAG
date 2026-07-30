import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

load_dotenv()

# Initialize Google GenAI Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Standard model identifier
MODEL_NAME = "gemini-3.5-flash"

def call_genai_with_retry(prompt, retries=3, backoff_factor=2):
    """Calls Gemini API with automatic retry logic for 503 errors."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            return response.text
        except APIError as e:
            # Check if it's a 503 error or temporary server error
            if e.code == 503 or "503" in str(e):
                if attempt < retries - 1:
                    sleep_time = backoff_factor ** attempt
                    time.sleep(sleep_time)
                    continue
            raise e
    raise Exception("Max retries exceeded for Gemini API call.")


class SimpleQAChain:
    """Wrapper class to handle document QA using Google GenAI SDK."""
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def invoke(self, inputs):
        query = inputs.get("query", "")
        
        # Retrieve relevant document chunks from FAISS
        docs = self.vector_store.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
You are an AI assistant that answers ONLY from the uploaded PDF context.

Rules:
1. Answer ONLY from the given context.
2. If the answer is not present, reply exactly:
   "I couldn't find this information in the uploaded PDFs."
3. Never make up information.
4. Keep the answer clear and professional.

Context:
{context}

Question:
{query}

Helpful Answer:
"""
        response_text = call_genai_with_retry(prompt)
        
        return {
            "result": response_text,
            "source_documents": docs
        }


def create_chatbot(vector_store):
    """Factory function returning the QA handler instance."""
    return SimpleQAChain(vector_store)


def generate_quiz_from_text(document_text, num_questions=5, difficulty="Medium"):
    """Generates a structured JSON quiz based on retrieved document content."""
    prompt = f"""
    Create a {num_questions}-question multiple-choice quiz based on the following document text.
    Target difficulty: {difficulty}.

    DOCUMENT CONTENT:
    {document_text[:12000]}

    STRICT OUTPUT FORMAT: Return ONLY a valid JSON array matching this format:
    [
        {{
            "id": 1,
            "question": "Question text?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "Option A",
            "explanation": "Explanation text"
        }}
    ]
    """
    
    raw_text = call_genai_with_retry(prompt).strip()
    
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
            
    try:
        return json.loads(raw_text.strip())
    except Exception:
        return None