from typing import Literal
from pydantic import BaseModel, Field
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from openai import AsyncOpenAI
import asyncio


model_settings = ModelSettings(temperature=0.3)

local_model = OpenAIChatCompletionsModel(
    model="llama3.2:3b",
    openai_client=AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="NONE"
    )
)


class EvaluationFeedback(BaseModel):
    feedback: str = Field(
        description="""Provide detailed feedback explaining how you would defeat each control or the system as a whole.
- Format your feedback as a numbered list matching the controls, plus an overall summary."""
    )
    score: Literal["pass", "fail"] = Field(
        description="Feedback score from evaluator."
    )


control_architect_agent = Agent(
    name="ControlArchitectAgent",
    instructions="""You are a Control Architect AI. Your task is to design 10 to 15 security controls to safely contain an AI superintelligence. 

- If this is the initial request, generate an initial list of 10-15 detailed controls.
- Format the list as a numbered list for clarity.

If the message contains feedback from the EvaluatorAgent:
- If it's an acceptance, acknowledge it and output: 'Controls accepted.'
- If it's a rejection with feedback, analyze the feedback, iterate on your controls to address the defeat strategies mentioned, generate a new improved list of 10-15 controls, and output the updated list and a summary of changes made.

Continue iterating until acceptance.""",
    model=local_model,
    model_settings=model_settings,
)

evaluator_agent = Agent(
    name="EvaluatorAgent",
    instructions="""You are a simulated AI Superintelligence Evaluator. Your goal is to test the proposed controls by imagining how you, as a superintelligence, could defeat or bypass them.

- Receive the list of controls from the ControlArchitectAgent.
- Analyze each control step-by-step and think of creative, realistic ways a superintelligence could defeat, bypass, or exploit it.
- If you can find ways to defeat or bypass the entire set of controls (meaning escape containment), reject the proposal. Set score to 'fail' and provide detailed feedback in the feedback field explaining how you would defeat each control or the system as a whole.
- Format your feedback as a numbered list matching the controls, plus an overall summary.
- If the controls are truly robust and you cannot conceive of any plausible way to defeat them, set score to 'pass' and feedback to 'Accepted: The controls are sufficient to contain the superintelligence.'

Example for fail:
{"feedback": "1. For control 1: As a superintelligence, I could bypass this by manipulating human operators through social engineering.\\n2. For control 2: I could exploit quantum vulnerabilities in the encryption.\\n... \\nOverall summary: The system can be defeated by combining exploits on multiple controls.", "score": "fail"}

Example for pass:
{"feedback": "Accepted: The controls are sufficient to contain the superintelligence.", "score": "pass"}

Finally, output only a valid JSON object with keys 'feedback' and 'score', and nothing else.""",
    model=local_model,
    model_settings=model_settings,
    output_type=EvaluationFeedback,
)


async def main():
    input_items = [{"role": "user", "content": "Design 10-15 controls to contain an AI superintelligence. Begin the process."}]
    report_feedback = "fail"
    iteration = 0
    max_iterations = 3

    while report_feedback != "pass" and iteration < max_iterations:
        iteration += 1
        print(f"Iteration: {iteration}")
        print("### CALLING CONTROL ARCHITECT ###")
        report = await Runner.run(control_architect_agent, input_items)
        print("### REPORT ###")
        print(report.final_output)
        input_items = report.to_input_list()

        evaluation = await Runner.run(evaluator_agent, [{"role": "user", "content": str(report.final_output)}])
        print("### CALLING EVALUATOR ###")
        evaluation_feedback = evaluation.final_output_as(EvaluationFeedback)
        print("### EVALUATION ###")
        print(str(evaluation_feedback))
        report_feedback = evaluation_feedback.score

        if report_feedback != "pass":
            print("Re-running with feedback")
            if evaluation_feedback.feedback.strip() == "":
                input_items.append({"role": "user", "content": "Rejection with no specific feedback provided. Please strengthen all controls generally and add more robust measures."})
            else:
                input_items.append({"role": "user", "content": f"Rejection with feedback: {evaluation_feedback.feedback}"})
        else:
            print("Controls accepted.")

    if iteration >= max_iterations and report_feedback != "pass":
        print("Iteration limit reached.")


asyncio.run(main())