import ollama

# Define a custom system prompt
system_prompt = "You are a helpful travel assistant that can customize travel plans to make for a great getaway."

# Prepare the message list with a system prompt
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Can you tell me about what is unique in Columbus, Ohio that I should do during my trip next week? "}
]

# Call the Ollama LLM with the custom system prompt
response = ollama.chat(
    model="llama3.2",
    messages=messages
)

print(response['message']['content'])


