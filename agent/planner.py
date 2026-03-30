"""
Planning functionality for the agent.

Planning is data generation, not reasoning.
Plans are inspectable, modifiable data structures.
"""

from shared.llm import LocalLLM


def create_plan(llm: LocalLLM, goal: str) -> dict | None:
    """
    Generate a plan to achieve a goal.
    
    Used in: Lesson 08
    
    Args:
        llm: The language model to use
        goal: The goal to achieve
        
    Returns:
        Plan as a dictionary with a "steps" list, or None if generation failed
    """
    from shared.utils import extract_json_from_text
    
    prompt = f"""Create a step-by-step plan to achieve the goal. Respond with ONLY valid JSON.

            CRITICAL INSTRUCTIONS:
            1. Respond with ONLY valid JSON
            2. No explanations, no markdown, no other text
            3. Start your response with {{ and end with }}
            
            Required JSON format:
            {{"steps": ["step1", "step2", "step3"]}}
            
            Goal: {goal}
            
            Response (JSON only):"""
    
    for attempt in range(3):
        response = llm.generate(prompt, temperature=0.0)
        plan = extract_json_from_text(response)
        
        if plan and "steps" in plan and isinstance(plan["steps"], list):
            return plan
    
    return None

def create_atomic_action(llm: LocalLLM, step: str) -> dict | None:
    """
    Convert a plan step into an atomic action.
    
    Used in: Lesson 09
    
    Args:
        llm: The language model to use
        step: A step from a plan
        
    Returns:
        Atomic action as a dictionary, or None if generation failed
    """
    from shared.utils import extract_json_from_text
    
    prompt = f"""Convert this step into an atomic action. Respond with ONLY valid JSON.

            CRITICAL INSTRUCTIONS:
            1. Respond with ONLY valid JSON
            2. No explanations, no markdown, no other text
            3. Start your response with {{ and end with }}
            
            Required JSON format:
            {{
              "action": "action_name",
              "inputs": {{"key": "value"}}
            }}
            
            The action should be a simple, atomic operation name.
            The inputs should be a dictionary with the parameters needed for this action.
            
            Step to convert:
            {step}
            
            Response (JSON only):"""
    
    for attempt in range(3):
        response = llm.generate(prompt, temperature=0.0)
        action = extract_json_from_text(response)
        
        if action and "action" in action:
            return action
    
    return None