import together
from together import Together
import os
from dotenv import load_dotenv
import tiktoken
import wandb
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Global table to store experiments
_experiments_table = None

def init_wandb():
    """
    Initialize wandb for experiment tracking.
    Should be called once at the start of experiments.
    """
    try:
        global _experiments_table
        
        run = wandb.init(
            project=os.getenv('WANDB_PROJECT', 'prompt-engineering-experiments'),
            entity=os.getenv('WANDB_ENTITY'),
            name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            },
            # Disable local storage
            settings=wandb.Settings(
                _disable_stats=True,
                mode="online"
            )
        )
        
        # Initialize the global table
        _experiments_table = wandb.Table(columns=[
            "timestamp",
            "experiment_type",
            "system_prompt",
            "user_prompt",
            "completion",
            "elapsed_time",
            "model",
            "run_id"
        ])
        
        print(f"Initialized wandb run: {run.name}")
        return True
        
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {str(e)}")
        print("Experiments will run without logging...")
        return False

def track_experiment(prompt, experiment_type, system_prompt=None, run_id=None):
    """
    Run and track an LLM experiment with wandb logging
    
    Args:
        prompt (str): The user prompt to send to the model
        experiment_type (str): Type of experiment (e.g., 'few-shot', 'role-based', etc.)
        system_prompt (str, optional): System prompt to prepend
        run_id (str, optional): Identifier for multiple runs of same prompt
    
    Returns:
        str: The model's completion
    """
    global _experiments_table
    
    # Time the experiment
    start_time = time.time()
    completion = llama(prompt, system_prompt=system_prompt)
    elapsed_time = time.time() - start_time
    
    # Log to wandb if available
    if wandb.run is not None and _experiments_table is not None:
        try:
            # Create a unique identifier for this run
            run_identifier = f"run_{run_id}" if run_id else datetime.now().strftime('%H%M%S_%f')
            
            # Format timestamp to be more readable
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Add row to the table
            _experiments_table.add_data(
                timestamp,  # More readable timestamp format
                experiment_type,
                system_prompt or "",
                prompt,
                completion,
                f"{elapsed_time:.2f}",  # Format elapsed time to 2 decimal places
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                run_identifier
            )
            
            # Log the updated table
            wandb.log({"experiments": _experiments_table})
            
            # Also log individual metrics
            wandb.log({
                "elapsed_time": elapsed_time,
                "prompt_length": len(prompt),
                "completion_length": len(completion),
                "experiment_type": experiment_type,
                "run_id": run_identifier
            })
            
        except Exception as e:
            print(f"Warning: Could not log to wandb: {str(e)}")
    
    return completion

def llama(prompt, system_prompt=None):
    """
    Get completion from Together AI's Llama model
    
    Args:
        prompt (str): The prompt to send to the model
        system_prompt (str, optional): System prompt to prepend
    
    Returns:
        str: The model's completion
    """
    # Initialize the Together client
    client = together.Together()
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Create the completion using the chat API
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
    )
    
    return response.choices[0].message.content