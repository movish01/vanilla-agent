from typing import Any
# from agent.agent import Agent
from shared.llm import LocalLLM

class Agent:
    def __init__(self, model_path: str):
        self.llm = LocalLLM(model_path)

    def simple_generate(self, user_input: str) -> str:
        """
        Generate a response based on the user input.
        """
        return self.llm.generate(user_input)