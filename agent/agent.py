from typing import Any
from shared.llm import LocalLLM
from shared.utils import extract_json_from_text

class Agent:
    def __init__(self, model_path: str):
        self.llm = LocalLLM(model_path)
        self.system_prompt = (
            "You are a helpful and precise assistant for answering questions.",
            "You explain concepts simply and clearly, and avoid unnecessary details.",
            "If you don't know the answer, say you don't know, don't try to make up an answer."
        )

    def simple_generate(self, user_input: str) -> str:
        """
        Generate a response based on the user input.
        """
        return self.llm.generate(user_input)
    
    def generate_with_role(self, user_input: str) -> str:
        """
        Generate a response with system prompt.
        """
        prompt = f"""{self.system_prompt}
        User: {user_input}
        Assistant:"""
        
        response = self.llm.generate(prompt)
        
        #Cleaning response
        response = response.replace('<SYSTEM>', '').replace('</SYSTEM>', '').strip()
        response = response.replace('<USER>', '').replace('</USER>', '').strip()
        return response
    
    def generate_structured(self, user_input: str, schema: str) -> dict | None:
        """Generate structure JSON outputwith vaidations

        Args:
            user_input (str): _description_
            schema (str): _description_

        Returns:
            dict | None: _description_
        """
        
        prompt = f"""{self.system_prompt}
            CRITICAL INSTRUCTIONS:
            1. Respond with ONLY JSON
            2. No explanations, no markdown, no extrac test before or after the JSON response
            3. Start your response with {{ and end with }}
            
            Schema you must follow:
            {schema}
            
            User request: {user_input}
            
            Reponse (JSON only):"""
        
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response) 
            
            if parsed is not None:
                return parsed
        
        return None