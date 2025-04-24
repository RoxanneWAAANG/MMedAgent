import os
import json
import random
import argparse
from dotenv import load_dotenv
from retry import retry
from tqdm import tqdm
import openai

# Load environment variables from .env
load_dotenv()

# System prompt for PMC-LLaMA QA task
system_message = (
    "You're a clinical question–answering assistant powered by the PMC-LLaMA model.\n"
    "When you receive a medical question, you MUST output exactly:\n"
    "Question: <the user's question>\n"
    "Answer:\n"
    "<thoughts> Why PMC-LLaMA is the right tool and how you'll retrieve evidence.\n"
    "<actions> [{'tool name':'PMC-LLaMA', 'tool params':{'query':'<the user's question>'}}]\n"
    "<values> <the final, evidence-based answer>\n"
)

# Pool of user question templates
user_templates = [
    "What are the primary risk factors for {disease}?",
    "How does {drug} work to treat {condition}?",
    "What is the long-term prognosis of a patient with {condition}?",
]

# Assistant template showing one shot format
assistant_template = (
    "Question: <question>\n\n"
    "Answer:\n\n"
    "<thoughts> I will search PubMed via PMC-LLaMA to gather authoritative sources.\n\n"
    "<actions> [{'tool name':'PMC-LLaMA','tool params':{'query':'<question>'}}]\n\n"
    "<values> <value>\n"
)

# Simple sampler for variables used in templates
def sample_vars():
    return {
        "disease": random.choice(["diabetes", "hypertension", "COPD"]),
        "drug": random.choice(["aspirin", "metformin", "lisinopril"]),
        "condition": random.choice(["heart failure", "asthma", "chronic kidney disease"]),
    }

# Retry wrapper for rate-limit errors
@retry(openai.error.RateLimitError, tries=3, delay=2, backoff=2)
def call_gpt4o(messages):
    response = openai.ChatCompletion.create(
        model="o4-mini-2025-04-16",
        messages=messages,
        temperature=0.8,
        max_tokens=512,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["message"]["content"]

# Build one instruction sample (3 warm-ups + final)
def process_one():
    vars = sample_vars()
    # Start with system message
    messages = [{"role": "system", "content": system_message}]
    # Add 3 warm-up examples
    for _ in range(3):
        question = random.choice(user_templates).format(**vars)
        assistant_shot = assistant_template.replace("<question>", question).replace("<value>", "…")
        messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_shot}
        ])
    # Final user query
    final_question = random.choice(user_templates).format(**vars)
    messages.append({"role": "user", "content": final_question})

    # Call the model to get the chain
    chain = call_gpt4o(messages)
    return {
        "task": "QA",
        "user": final_question,
        "chain": chain
    }

# Main entry: generate N examples and save to JSON
def main(output_path: str, total_num: int):
    # Set API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")

    results = []
    for _ in tqdm(range(total_num), desc="Generating QA examples"):
        results.append(process_one())

    # Write output file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA instruction data using PMC-LLaMA.")
    parser.add_argument("--output", type=str, default="qa_instruct_pmcllama.json", help="Output JSON file path.")
    parser.add_argument("--total_num", type=int, default=50, help="Number of examples to generate.")
    args = parser.parse_args()
    main(args.output, args.total_num)
