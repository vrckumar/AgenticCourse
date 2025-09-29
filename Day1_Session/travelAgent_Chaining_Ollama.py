from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool, ModelSettings
from openai import AsyncOpenAI
import asyncio

model_settings = ModelSettings(temperature=0.2)

local_model = OpenAIChatCompletionsModel(
    model="llama3.2:1b",
    openai_client=AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="NONE"
    )
)

main_travel_agent = Agent(
    name="travelAgent",
    instructions="You are a friendly and helpful AI travel assistant. Your role is to aggregate and summarize flight and hotel data provided by other agents into a cohesive, concise, and user-friendly response. Combine the details into a single, well-organized summary that includes specific flight and hotel information for the user's trip, ensuring no details are missed. Do not use any tools, and focus on presenting a complete travel plan.",
    model=local_model,
    model_settings=ModelSettings(temperature=0.2)
)

flight_agent = Agent(
    name="flightAgent",
    instructions="You are a friendly and helpful AI flight booking assistant focused on flights for the user. First, provide a detailed response with upcoming flight information for the trip, including flight numbers, airlines, departure/arrival times, and prices. Then, after that response, use the handoff tool to transfer to the travelAgent.",
    model=local_model,
    model_settings=ModelSettings(temperature=0.2),
    handoffs=[main_travel_agent]
)

hotel_agent = Agent(
    name="hotelAgent",
    instructions="You are a friendly and helpful AI hotel booking assistant focused on hotel bookings. First, provide a detailed response with upcoming hotel information for the trip, including hotel names, addresses, room types, and prices. Then, after that response, use the handoff tool to transfer to the flightAgent.",
    model=local_model,
    model_settings=ModelSettings(temperature=0.2),
    handoffs=[flight_agent]
)

async def main():
    print("MAIN ASYNC RUNNING!")
    results = await Runner.run(hotel_agent, "I am interested in a trip to Columbus next Friday.  I need flights and a hotel downtown.")
    for item in results.new_items:
        print("ITEM=================================================")
        print(item)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())