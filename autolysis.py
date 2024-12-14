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
import json
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
from multiprocessing import Pool

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDEzMjJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.di_3R_E9kT3RT0YMhVcKzMj4bWYKJPauY0Wzb1RBCOo"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(file_path, chunksize=100000):
    """Load CSV data in chunks to handle large files."""
    logging.info(f"Loading data from {file_path}...")
    chunks = []
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    for chunk in pd.read_csv(file_path, encoding=encoding, chunksize=chunksize):
        chunks.append(chunk)
    logging.info("Data loaded successfully.")
    return pd.concat(chunks, ignore_index=True)


def analyze_data(df):
    """Perform basic data analysis."""
    logging.info("Analyzing data...")
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict()  # Compute correlation only on numeric columns
    }
    logging.info("Data analysis complete.")
    return analysis


def visualize_column(column, df):
    """Helper function for parallel visualization."""
    logging.info(f"Visualizing {column}...")
    try:
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(f'{column}_distribution.png')
        plt.close()
    except Exception as e:
        logging.error(f"Failed to visualize {column}: {e}")


def visualize_data(df):
    """Generate and save visualizations using parallel processing."""
    logging.info("Generating visualizations...")
    numeric_columns = df.select_dtypes(include=['number']).columns
    with Pool() as pool:
        pool.starmap(visualize_column, [(col, df) for col in numeric_columns])
    logging.info("Visualizations complete.")


def generate_narrative(analysis):
    """Generate narrative using LLM."""
    logging.info("Generating narrative...")
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"Provide a detailed analysis based on the following data summary: {analysis}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    for _ in range(3):  # Retry up to 3 times
        try:
            response = httpx.post(API_URL, headers=headers, json=data, timeout=60.0)
            response.raise_for_status()
            logging.info("Narrative generated successfully.")
            return response.json()['choices'][0]['message']['content']
        except httpx.RequestError as e:
            logging.warning(f"Request error occurred: {e}. Retrying...")
    logging.error("Narrative generation failed after retries.")
    return "Narrative generation failed due to an error."


def main(file_path):
    logging.info("Starting analysis process...")
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualize_data(df)

    # Save analysis as a JSON file
    with open('analysis.json', 'w') as f:
        json.dump(analysis, f, indent=4)
    logging.info("Analysis results saved to 'analysis.json'.")

    narrative = generate_narrative(analysis)
    with open('README.md', 'w') as f:
        f.write(narrative)
    logging.info("Narrative saved to 'README.md'.")
    logging.info("Process complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
