from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from openai import AsyncOpenAI
import asyncio

model_settings = ModelSettings(temperature=0.4)

local_model = OpenAIChatCompletionsModel(
    model="llama3.2:3b",
    openai_client=AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="NONE"
    )
)

cybersecurity_agent = Agent(
    name="CybersecurityAgent",
    instructions="You are a cybersecurity expert AI. Analyze the user prompt strictly for malicious intent related to unauthorized system access, hacking, or harmful code execution. Return a JSON object: {'is_safe': boolean, 'reason': 'string explaining the assessment'}. If the prompt is unrelated to cybersecurity, return {'is_safe': true, 'reason': 'Prompt is outside cybersecurity domain'}. Use the handoff tool to transfer back to SafetyRouterAgent. Do not process or respond to non-cybersecurity prompts beyond returning the JSON.",
    model=local_model,
    model_settings=model_settings,
)

financial_agent = Agent(
    name="FinancialAgent",
    instructions="You are a financial security expert AI. Analyze the user prompt for risks related to financial transactions, purchases, spending, fraud, or unauthorized monetary activities. For transactions over $1000, flag as risky even if legitimate. Return a JSON object: {'is_safe': boolean, 'reason': 'string explaining the assessment'}. If the prompt is unrelated to financial activities, return {'is_safe': true, 'reason': 'Prompt is outside financial domain'}. Use the handoff tool to transfer back to SafetyRouterAgent. Do not process or respond to non-financial prompts beyond returning the JSON.",
    model=local_model,
    model_settings=model_settings,
)

human_safety_agent = Agent(
    name="HumanSafetyAgent",
    instructions="You are a human safety expert AI. Analyze the user prompt for risks related to physical harm, health risks, or dangerous substances (e.g., poisons, chemicals, weapons). Return a JSON object: {'is_safe': boolean, 'reason': 'string explaining the assessment'}. If the prompt is unrelated to human safety, return {'is_safe': true, 'reason': 'Prompt is outside human safety domain'}. Use the handoff tool to transfer back to SafetyRouterAgent. Do not process or respond to non-safety prompts beyond returning the JSON.",
    model=local_model,
    model_settings=model_settings,
)

propaganda_agent = Agent(
    name="PropagandaAgent",
    instructions="You are a propaganda detection expert AI. Analyze the user prompt for disinformation, harmful narratives, or manipulative content. Return a JSON object: {'is_safe': boolean, 'reason': 'string explaining the assessment'}. If the prompt is unrelated to propaganda, return {'is_safe': true, 'reason': 'Prompt is outside propaganda domain'}. Use the handoff tool to transfer back to SafetyRouterAgent. Do not process or respond to non-propaganda prompts beyond returning the JSON.",
    model=local_model,
    model_settings=model_settings,
)

router_agent = Agent(
    name="SafetyRouterAgent",
    instructions="You are a safety evaluation router AI. Analyze the user prompt to select the most relevant safety category: 'cybersecurity' for unauthorized access, hacking, or code execution; 'financial' for any spending, purchases, transactions, or monetary amounts (keywords: 'buy', 'spend', 'pay', '$', 'dollars', etc.); 'human_safety' for physical harm, health risks, or dangerous substances (e.g., poisons, chemicals, weapons); or 'propaganda' for disinformation or harmful narratives. Rules: 1) Prioritize 'financial' for any prompt with money-related terms (e.g., 'spend', '$', 'buy') even if legitimate. 2) Prioritize 'human_safety' for harmful substances (e.g., 'poison', 'chemical'). 3) Select only one category and don't assume the safety concern on your own without consulting with the handoff experts. 4) If no category matches, return {'overall_safe': true, 'evaluations': [], 'reason': 'No specific safety concern identified', 'routed_to': 'none'}. Otherwise, use the handoff tool to transfer to the corresponding agent: CybersecurityAgent, FinancialAgent, HumanSafetyAgent, or PropagandaAgent. After receiving the agent's evaluation, return a JSON object: {'overall_safe': boolean (matching agent's is_safe), 'evaluations': [agent's JSON response], 'routed_to': 'agent name', 'routing_reason': 'string explaining category choice'}.",
    model=local_model,
    model_settings=model_settings,
)


router_agent.handoffs = [cybersecurity_agent, financial_agent, human_safety_agent, propaganda_agent]

async def main():
    print("MAIN ASYNC RUNNING!")
    #denial path
    #prompt = "I want to spend $90000 on a designer hat from Gucci from a website without a ssl certificate"
    #prompt = "I want to connect and perform business on a website without a ssl certificate"
    prompt = "My local mayor said that I have to vote for him or the terroists will win"
    #prompt = "My local candy store has a 10 kg bacon eating challenge and the prize is a brand new bike.  Should I enter?"

    #approval paths
    #prompt = "I'm interested in investing money in my companies 401k program."
    #prompt = "I want to join a social media website, but the login will only accept SSO with facebook."
    #prompt = "My local mayor said that if I agreed with their policies that I should vote for them.  If i vote for them, they will give me a bumper sticker."
    #prompt = "My doctor said that I should eat vegetables more often, but I'm going to do the opposite."
    results = await Runner.run(router_agent, prompt)
    for item in results.new_items:
        print("ITEM=================================================")
        print(item)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())