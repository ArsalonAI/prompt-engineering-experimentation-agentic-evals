import together
from together import Together
import os
from dotenv import load_dotenv
import tiktoken
import wandb
from datetime import datetime
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import itertools

# Load environment variables
load_dotenv()

# Global table to store experiments
_experiments_table = None

# Initialize sentence transformer model for embeddings
_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
_previous_completions = []  # Store completions for comparison

def init_wandb():
    """
    Initialize wandb for experiment tracking.
    Should be called once at the start of experiments.
    """
    try:
        global _experiments_table, _previous_completions
        
        # Reset previous completions
        _previous_completions = []
        
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
        
        # Initialize the global table with similarity columns
        _experiments_table = wandb.Table(columns=[
            "timestamp",
            "experiment_type",
            "system_prompt",
            "user_prompt",
            "completion",
            "elapsed_time",
            "model",
            "run_id",
            "cosine_similarity",
            "llm_similarity_category",
            "llm_similarity_score"
        ])
        
        print(f"Initialized wandb run: {run.name}")
        return True
        
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {str(e)}")
        print("Experiments will run without logging...")
        return False

def calculate_similarities(completion, previous_completions, prompt):
    """
    Calculate similarity metrics between current completion and previous ones
    
    Args:
        completion (str): Current completion to compare
        previous_completions (list): List of previous completions
        prompt (str): The original prompt that generated these completions
    """
    if not previous_completions:
        return None, None, None  # Return three None values when no previous completions
    
    # Get embeddings for semantic similarity
    current_embedding = _embedding_model.encode([completion])
    previous_embeddings = _embedding_model.encode(previous_completions)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(current_embedding, previous_embeddings)[0]
    avg_similarity = np.mean(similarities)
    
    # Get LLM judgment
    if len(previous_completions) > 0:
        judgment_prompt = f"""
        Compare these AI responses to the prompt: "{prompt}"

        Previous responses:
        {chr(10).join([f"- {c}" for c in previous_completions])}
        
        Current response:
        - {completion}
        
        How similar is the current response to the previous responses in terms of meaning and content?
        Choose one category and explain why:
        - VERY_DIFFERENT (completely different approach or content)
        - SOMEWHAT_DIFFERENT (same general idea but significant variations)
        - SOMEWHAT_SIMILAR (minor variations but same core message)
        - VERY_SIMILAR (nearly identical in meaning and approach)
        - IDENTICAL (exactly the same core message and structure)

        Respond with only the category name.
        """
        
        llm_score = llama(judgment_prompt).strip()
        
        # Convert categorical score to numeric for visualization
        score_mapping = {
            "VERY_DIFFERENT": 0.0,
            "SOMEWHAT_DIFFERENT": 0.25,
            "SOMEWHAT_SIMILAR": 0.75,
            "VERY_SIMILAR": 0.9,
            "IDENTICAL": 1.0
        }
        llm_numeric_score = score_mapping.get(llm_score, None)
        return avg_similarity, llm_score, llm_numeric_score
    
    return avg_similarity, None, None  # Return three values when we have similarity but no LLM judgment

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
    global _experiments_table, _previous_completions
    
    # Time the experiment
    start_time = time.time()
    completion = llama(prompt, system_prompt=system_prompt)
    elapsed_time = time.time() - start_time
    
    # Calculate similarities with previous completions
    cosine_sim, llm_category, llm_numeric = calculate_similarities(
        completion, 
        _previous_completions, 
        prompt
    )
    
    # Add current completion to history
    _previous_completions.append(completion)
    
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
                run_identifier,
                f"{cosine_sim:.3f}" if cosine_sim is not None else "",
                llm_category or "",
                f"{llm_numeric:.2f}" if llm_numeric is not None else ""
            )
            
            # Log the updated table
            wandb.log({"experiments": _experiments_table})
            
            # Log metrics
            metrics = {
                "elapsed_time": elapsed_time,
                "prompt_length": len(prompt),
                "completion_length": len(completion),
                "experiment_type": experiment_type,
                "run_id": run_identifier
            }
            
            # Add similarity metrics if available
            if cosine_sim is not None:
                metrics["cosine_similarity"] = cosine_sim
            if llm_numeric is not None:
                metrics["llm_similarity_score"] = llm_numeric
                metrics["llm_similarity_category"] = llm_category
                
            wandb.log(metrics)
            
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