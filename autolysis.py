# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "numpy",
#   "ipykernel"
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = "your_token_here"

# Helper Function for Dynamic Prompt Generation
def create_prompt(analysis, visualizations):
    """Generate a structured prompt for the LLM based on analysis and visualizations."""
    return f"""
    Create a Markdown report that includes:
    1. Overview of the dataset.
    2. Key insights from the following analysis:
       - Summary statistics: {analysis['summary']}
       - Missing values: {analysis['missing_values']}
       - Key correlations: {list(analysis['correlation'].items())[:5]}
    3. Observations based on visualizations:
       {visualizations}
    Include actionable implications for decision-making and highlight significant findings.
    """

# Load data
def load_data(file_path):
    """Load CSV data with encoding detection."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        logging.info(f"Detected encoding: {encoding}")
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)

# Data analysis
def analyze_data(df):
    """Perform dynamic data analysis."""
    try:
        numeric_df = df.select_dtypes(include=['number'])
        analysis = {
            'summary': df.describe(include='all').to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'correlation': numeric_df.corr().unstack().sort_values(ascending=False).drop_duplicates().head(5).to_dict()
        }
        logging.info("Data analysis completed successfully.")
        return analysis
    except Exception as e:
        logging.error(f"Data analysis failed: {e}")
        sys.exit(1)

# Visualization generation
def visualize_data(df):
    """Generate and save visualizations."""
    visualizations = []
    try:
        sns.set(style="whitegrid")
        numeric_columns = df.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            plt.figure()
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            file_name = f'{column}_distribution.png'
            plt.savefig(file_name)
            plt.close()
            visualizations.append(f"Visualization of {column}: {file_name}")
        logging.info("Visualizations created successfully.")
    except Exception as e:
        logging.error(f"Visualization generation failed: {e}")
    return visualizations

# Generate narrative using LLM
def generate_narrative(analysis, visualizations):
    """Generate narrative using LLM with structured prompt."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = create_prompt(analysis, visualizations)
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    retries = 3
    for attempt in range(retries):
        try:
            response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            logging.info("Narrative generation successful.")
            return response.json()['choices'][0]['message']['content']
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {e}")
        except httpx.RequestError as e:
            logging.error(f"Request error occurred: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
        logging.info(f"Retrying narrative generation ({attempt + 1}/{retries})...")
        time.sleep(2)

    return "Narrative generation failed after multiple attempts."

# Main workflow
def main(file_path):
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualizations = visualize_data(df)
    narrative = generate_narrative(analysis, visualizations)
    
    with open('README.md', 'w') as f:
        f.write(narrative)
    logging.info("Process completed and README.md generated.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
