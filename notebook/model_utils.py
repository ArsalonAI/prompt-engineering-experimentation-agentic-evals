import together
import os
from dotenv import load_dotenv
import tiktoken


def llama(prompt):
    """
    Helper function to interact with Together AI's Llama model
    
    Args:
        prompt (str): The input prompt for the model
        
    Returns:
        str: The generated response
    """
    # Initialize Together client
    client = together.Together()
    
    # Create the completion
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=None,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"]
    )
    
    # Extract and return the response text
    return response.choices[0].message.content