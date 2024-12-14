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
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDEzMjJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.di_3R_E9kT3RT0YMhVcKzMj4bWYKJPauY0Wzb1RBCOo"

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

def analyze_data(df):
    """Perform basic data analysis."""
    try:
        numeric_df = df.select_dtypes(include=['number'])
        analysis = {
            'summary': df.describe(include='all').to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'correlation': numeric_df.corr().to_dict()
        }
        logging.info("Data analysis completed successfully.")
        return analysis
    except Exception as e:
        logging.error(f"Data analysis failed: {e}")
        sys.exit(1)

def visualize_data(df):
    """Generate and save visualizations."""
    try:
        sns.set(style="whitegrid")
        numeric_columns = df.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            plt.figure()
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f'Distribution of {column}')
            plt.savefig(f'{column}_distribution.png')
            plt.close()
        logging.info("Visualizations created successfully.")
    except Exception as e:
        logging.error(f"Visualization generation failed: {e}")

def generate_narrative(analysis):
    """Generate narrative using LLM with retry mechanism."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"Provide a detailed analysis based on the following data summary: {analysis}"
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

def main(file_path):
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualize_data(df)
    narrative = generate_narrative(analysis)
    with open('README.md', 'w') as f:
        f.write(narrative)
    logging.info("Process completed and README.md generated.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
