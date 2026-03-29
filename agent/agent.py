from typing import Any
from shared.llm import LocalLLM
from shared.utils import extract_json_from_text
from agent.tools import get_tool_schema, execute_tool
from agent.state import AgentState

class Agent:
    def __init__(self, model_path: str):
        self.llm = LocalLLM(model_path)
        self.system_prompt = (
            "You are a helpful and precise assistant for answering questions.",
            "You explain concepts simply and clearly, and avoid unnecessary details.",
            "If you don't know the answer, say you don't know, don't try to make up an answer."
        )
        self.state = AgentState()

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
    
    def decide(self, user_input: str, options: list[str]) -> str | None:
        """The model should decide from a finite set of options.

        Args:
            user_input (str): The input to make a decision about 
            options (list[str]): The finite set of options to choose from
        
        Returns:
            The chosen option, or None if no valid option was chosen.
        """
        option_list = "\n".join(f"- {option}" for option in options)
        prompt = f"""{self.system_prompt}
            CRITICAL INSTRUCTIONS:
            1. Reponsd with ONLY valid JSON
            2. No explanations, no markdown, no extra text before or after the JSON response
            3. Start your response with {{ and end with }}
            
            Available options:
            {option_list}
            
            Requied JSON format:
            {{"decision": "one of the options above"}}
            
            User request: {user_input}
            
            Response (JSON only):"""
            
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response)
            
            if parsed is not None and "decision" in parsed and parsed["decision"] in options:
                decision = parsed["decision"]
                if decision in options:
                    return decision
        
        return None

    def request_tool(self, user_input: str) -> dict | None:
        """
        Let the model reques a tool call.

        Args:
            user_input (str): The user's request

        Returns:
            dict | None: The tool call request details, or None if no valid request was made.
        """
        prompt = f"""{self.system_prompt}
        
        You are a tool-calling assistant, When asked a math question, you must respond wiht ONLY valid JSON.
        
        Available tool: calculator
        - Parameters: a (number), b (number), operation (string: "add", "subtract", "multiply", "divide")
        
        CRITICAL INSTRUCTIONS:
        1. Respond with ONLY valid JSON
        2. No explanations, no markdown, no extra text before or after the JSON response
        3. Start your response with {{ and end with }}
        
        Example format:
        {{"tool" : "calculator", "arguments": {{"a": 5, "b": 3, "operation": "add"}}}}
        
        User request: {user_input}
        
        Response (JSON only):"""
        
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response)
            
            if parsed and "tool" in parsed and "arguments" in parsed:
                return parsed
        
        return None
    
    def execute_tool_call(self, tool_call: dict) -> Any:
        """
        Execute a tool call based on the model's request.

        Args:
            tool_call (dict): The tool call details, including the tool name and parameters.

        Returns:
            The result of the tool execution.
        """
        return execute_tool(tool_call["tool"], tool_call["arguments"])

    def agent_step(self, user_input: str) -> dict | None:
        """
        Model will execute one step of the agent loop/

        Args:
            user_input (str): User's input or system observations

        Returns:
            The result of the agent's step, or None if no action was taken.
        """
        state_dict = self.state.to_dict()
        prompt = f"""{self.system_prompt}
        
        You are an agent. You must decide the next action and respond with ONLY valid JSON.
        
        Current state: steps={state_dict.get('steps', 0)}, done={state_dict.get('done', False)}
        
        Available actions: analyze, research, sumarize, answer, done
        
        CRITICAL INSTRUCTIONS:
        1. Respond with ONLY valid JSON
        2. No explanations, no markdown, no extra text before or after the JSON response
        3. Start your response with {{ and end with }}
        
        Required JSON format:
        {{"action": "action_name", "reason": "explanation"}}
        
        User input: {user_input}
        
        Response (JSON only):"""
        
        for attempt in range(3):
            response = self.llm.generate(prompt, temperature=0.0)
            parsed = extract_json_from_text(response)
            
            if parsed and "action" in parsed:
                if "reason" not in parsed:
                    parsed["reason"] = f"Taking action: {parsed['action']}"
                self.state.increment_step()
                return parsed
        return None
    
    def run_loop(self, user_input: str, max_steps: int = 5):
        """
        Run the agent loop for multiple steps.

        Args:
            user_input (str): Initial user input
            max_steps (int): Maximum number of steps to execute. Defaults to 5.
        """
        self.state.reset()
        results = []
        
        while not self.state.done and self.state.steps < max_steps:
            action = self.agent_step(user_input)
            
            if action:
                results.append(action)
                
                if action.get("action") == "done":
                    self.state.mark_done()
            else:
                break
        
        return results