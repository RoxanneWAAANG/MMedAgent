import os
import json
import argparse
from dotenv import load_dotenv
from retry import retry
from tqdm import tqdm
from openai import OpenAI
import re

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt for PMC-LLaMA QA task
system_message = (
    "You're a clinical QA assistant using PMC-LLaMA.\n"
    "Follow this format exactly:\n"
    "Question: <the user's question>\n"
    "Answer:\n"
    "<thoughts> your reasoning for using PMC-LLaMA\n"
    "<actions> [{'tool name':'PMC-LLaMA','tool params':{'query':'<question>'}}]\n"
    "<values> <the answer text>"
)

@retry(Exception, tries=3, delay=2, backoff=2)
def call_model(messages):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.8,
        max_tokens=512
    )
    return resp.choices[0].message.content

# Build one conversation entry from a QA pair
def process_one(idx, entry):
    question = entry['input'].strip()
    # Assemble messages: system + one-shot
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question}
    ]
    # Call the model to get chain
    chain = call_model(messages)

    # Parse chain into components
    thoughts_match = re.search(r"<thoughts>(.*?)<actions>", chain, re.S)
    actions_match = re.search(r"<actions>(.*?)<values>", chain, re.S)
    values_match = re.search(r"<values>(.*)", chain, re.S)
    thoughts = thoughts_match.group(1).strip() if thoughts_match else ""
    actions_str = actions_match.group(1).strip() if actions_match else "[]"
    try:
        actions = json.loads(actions_str.replace("'", '"'))
    except:
        actions = []
    answer = values_match.group(1).strip() if values_match else ""

    # Simulate tool output placeholder
    tool_output = {"answer": answer}

    return {
        "id": idx,
        "instruction": entry.get('instruction', ''),
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "thoughts": thoughts, "actions": actions, "value": "Calling PMC-LLaMA..."},
            {"from": "human", "value": f"PMC-LLaMA output: {json.dumps(tool_output)}\n\n"},
            {"from": "gpt", "thoughts": "Based on the retrieved answer, here is the medical advice.", "actions": [], "value": answer}
        ]
    }

# Main: read QA pairs from file and generate output
def main(input_path, output_path, total):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")

    # Load QA entries
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Limit to total entries if specified
    data = data[:total]

    results = []
    for i, entry in enumerate(tqdm(data, desc="Processing QA pairs")):
        results.append(process_one(i, entry))

    # Write output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    p = argparse.ArgumentParser(description="Generate PMC-LLaMA Instruction Turning Dataset.")
    p.add_argument("--input", required=True, help="Path to JSON file with QA pairs.")
    p.add_argument("--output", required=True, help="Output JSON file path.")
    p.add_argument("--total_num", type=int, default=50, help="Number of examples to process.")
    args = p.parse_args()
    main(args.input, args.output, args.total_num)
