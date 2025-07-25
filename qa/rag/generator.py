from dotenv import load_dotenv
import openai
import os
import re

load_dotenv()  # Load environment variables from .env

def format_as_points(answer):
    # If answer already has line breaks or is short, return as is
    if '\n' in answer or len(answer) < 120:
        return answer
    # Split into sentences
    sentences = re.split(r'(?<=[.!?]) +', answer)
    # If only one sentence, return as is
    if len(sentences) <= 1:
        return answer
    # Format as numbered list
    points = [f'{i+1}. {s.strip()}' for i, s in enumerate(sentences) if s.strip()]
    return '\n'.join(points)

def generate_answer(original_answer, query):
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise Exception('OPENAI_API_KEY not set in environment')
    client = openai.OpenAI(api_key=api_key)
    system_prompt = (
        "You are a helpful assistant. Here is an answer from our knowledge base:\n\n"
        f"{original_answer}\n\n"
        "Please refine and improve the answer above for clarity, completeness, and helpfulness, but do NOT omit any information. "
        "You may rephrase, elaborate, or format the answer, but all original information must be preserved."
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=256,
        temperature=0.0,  # Less creativity, more context-bound
    )
    llm_answer = response.choices[0].message.content.strip()
    return format_as_points(llm_answer) 