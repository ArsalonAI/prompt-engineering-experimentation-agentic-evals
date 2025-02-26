#!/bin/bash

# Create virtual environment
python -m venv csc7901_env

# Activate virtual environment
<<<<<<< HEAD
source csc7901_env/bin/activate  # On Windows, use: csc7901_env\Scripts\activate
=======
source csc7901_env/bin/activate  
>>>>>>> temp_branch

# Install requirements
pip install -r requirements.txt

# Install IPython kernel for Jupyter
python -m ipykernel install --user --name=csc7901_env --display-name="CSC7901 AI Experiments" 