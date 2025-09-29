import ollama

# Load the .txt file
with open('CBUS2025.txt', 'r', encoding='utf-8') as file:
    document_text = file.read()

# System prompt including the full document
system_prompt = f"You are a helpful travel assistant that can customize travel plans to make for a great getaway. Exclusively use the following information to inform your response: {document_text}"

# User query
user_query = "What is unique in Columbus, Ohio that I should do during my trip next week?"

# Prepare messages
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
]

# Call Ollama
response = ollama.chat(model="llama3.2:1b", messages=messages)
print(response['message']['content'])