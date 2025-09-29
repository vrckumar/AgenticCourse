import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from openai import AsyncOpenAI

model_settings = ModelSettings(temperature=0.4)
local_model = OpenAIChatCompletionsModel(
    model="granite3-moe:1b",
    openai_client=AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="NONE")
)

flight_instructions = "You are a flight booking assistant. Given a user query about flights, return a JSON object with flight details: flight_number (string), airline (string), departure_time (string, ISO format), arrival_time (string, ISO format), price (number). Ensure valid JSON only."

hotel_instructions = "You are a hotel booking assistant. Given a user query about hotels, return a JSON object with hotel details: hotel_name (string), address (string), room_type (string), price_per_night (number). Ensure valid JSON only."

flight_agent = Agent(
    name="flightAgent",
    instructions=flight_instructions,
    model=local_model,
    model_settings=model_settings
)

hotel_agent = Agent(
    name="hotelAgent",
    instructions=hotel_instructions,
    model=local_model,
    model_settings=model_settings
)

description_flight = "Retrieve flight details for the user's trip query."
description_hotel = "Retrieve hotel details for the user's trip query."

flight_tool = flight_agent.as_tool(tool_name="get_flight_details", tool_description=description_flight)
hotel_tool = hotel_agent.as_tool(tool_name="get_hotel_details", tool_description=description_hotel)

tools = [flight_tool, hotel_tool]

instructions = "You are a travel assistant. Analyze the user query to determine if flights, hotels, or both are needed. Call the appropriate tools (get_flight_details and/or get_hotel_details) with the full query as input to get JSON details. Then, create a concise, user-friendly summary of the travel plan, including all details. If a service is not requested, note it. Always use tools for details; do not invent information."

travel_agent = Agent(
    name="Travel Agent",
    instructions=instructions,
    tools=tools,
    model=local_model,
    model_settings=model_settings
)

async def main():
    query = "I am interested in a trip to Columbus next Friday, coming in from NYC. I need flights and a hotel downtown."
    result = await Runner.run(travel_agent, query)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())