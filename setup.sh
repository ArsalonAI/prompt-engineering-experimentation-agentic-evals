#!/bin/bash

# Create virtual environment
python -m venv csc7901_env

# Activate virtual environment
source csc7901_env/bin/activate  # On Windows, use: csc7901_env\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install IPython kernel for Jupyter
python -m ipykernel install --user --name=csc7901_env --display-name="CSC7901 AI Experiments" 