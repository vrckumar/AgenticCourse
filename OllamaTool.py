import ollama
import json
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

# Define the tool function to fetch real weather data
def get_current_weather(city: str) -> dict:
    try:
        # Fetch JSON from wttr.in (free public weather service)
        print(city)
        url = f"https://wttr.in/{city}?format=j1"
        with urlopen(url) as response:
            data = json.loads(response.read().decode())
        #print(data)
        
        
        # Extract relevant info from the response
        current = data.get('current_condition', [{}])[0]
        #print(current)
        return {
            "temperature": current.get('temp_C', 'N/A'),
            "condition": current.get('weatherDesc', [{}])[0].get('value', 'N/A')
        }
    except (HTTPError, URLError, json.JSONDecodeError, IndexError):
        return {"error": "Failed to fetch weather data"}

# Define the tool schema (JSON schema for the model to understand)
weather_tool = {
    'type': 'function',
    'function': {
        'name': 'get_current_weather',
        'description': 'Get the current weather for a city',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'The name of the city',
                },
            },
            'required': ['city'],
        },
    },
}

# Map function names to callable functions
available_functions = {
    'get_current_weather': get_current_weather,
}
# Define a custom system prompt
system_prompt = "You are a agent that can use tools to get information about the weather in a city.  Your answer to a city's weather should consist of the day of week and the high and low temperatures only, and the response should be given in plain english."

# Prepare the message list with a system prompt
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather in Columbus for next Tuesday?"}

]

# First chat: Model decides if/when to call tools
response = ollama.chat(
    model='llama3.1',
    messages=messages,
    tools=[weather_tool],
    options={"temperature":0.1}
)

# Check for tool calls
if 'tool_calls' in response['message']:
    messages.append(response['message'])  # Add the model's response to history
    
    for tool_call in response['message']['tool_calls']:
        function_name = tool_call['function']['name']
        function_args = tool_call['function']['arguments']
        
        if function_to_call := available_functions.get(function_name):
            # Execute the tool
            function_result = function_to_call(**function_args)
            
            # Add tool result back as a 'tool' message
            messages.append({
                'role': 'tool',
                'name': function_name,
                'content': json.dumps(function_result),
            })
    
    # Second chat: Send updated messages for the model to generate final answer
    final_response = ollama.chat(
        model='llama3.1',
        messages=messages,
    )
    print(final_response)
    print(final_response['message']['content'])
else:
    # No tool needed, just print the response
    print(response['message']['content'])