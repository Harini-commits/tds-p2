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
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN", "your_default_token")

# Helper Function for Dynamic Prompt Generation
def create_prompt(analysis, visualizations):
    """Generate a structured prompt for the LLM based on analysis and visualizations."""
    return f"""
    Create a detailed Markdown report based on the provided analysis and visualizations. The report should include:
    
    # Dataset Overview
    - A summary of the dataset's key characteristics and dimensions.
    
    # Data Analysis
    - Summary statistics: {analysis['summary']}
    - Missing values: {analysis['missing_values']}
    - Top correlations: {list(analysis['correlation'].items())[:5]}
    
    # Visualization Insights
    - Observations from visualizations:
    {"\n".join(visualizations)}
    
    # Key Findings and Implications
    - Highlight the most significant trends or patterns in the data.
    - Discuss actionable insights or implications for decision-making.
    
    Ensure that the report uses Markdown formatting with headings, bullet points, and tables where appropriate.
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
            sns.histplot(df[column].dropna(), kde=True, color='blue', bins=20)
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
def call_llm(prompt):
    """Make an LLM call with retries."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    retries = 3
    for attempt in range(retries):
        try:
            response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error in LLM call: {e}")
            time.sleep(2)
    return "Failed to generate insights after multiple attempts."

# Generate insights for visualizations
def generate_visual_summary(column, file_name):
    """Generate insights for a specific visualization."""
    prompt = f"""
    Analyze the distribution for {column} based on the visualization saved as {file_name}. 
    Highlight key trends, anomalies, and implications for this data.
    """
    return call_llm(prompt)

# Generate insights for correlations
def generate_correlation_summary(correlations):
    """Generate insights for top correlations."""
    prompt = f"""
    Summarize the following key correlations in the dataset: {correlations}.
    Discuss potential implications and relationships.
    """
    return call_llm(prompt)

# Generate final Markdown narrative
def generate_final_narrative(dataset_summary, visual_summary, correlation_summary):
    """Combine all sections into a Markdown narrative."""
    prompt = f"""
    Create a Markdown report with the following sections:
    1. Dataset Overview: {dataset_summary}.
    2. Key Correlation Insights: {correlation_summary}.
    3. Visualization Analysis: {visual_summary}.
    Highlight actionable insights and decision-making implications.
    """
    return call_llm(prompt)

# Main workflow
def main(file_path):
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualizations = visualize_data(df)

    # Generate separate insights
    visual_summaries = [generate_visual_summary(col, f"{col}_distribution.png") for col in df.select_dtypes(include='number').columns]
    correlation_summary = generate_correlation_summary(analysis['correlation'])
    dataset_summary = call_llm(f"Summarize the dataset: {analysis['summary']}")

    # Combine everything into a final narrative
    final_narrative = generate_final_narrative(dataset_summary, visual_summaries, correlation_summary)

    # Save the final Markdown report
    with open('README.md', 'w') as f:
        f.write(final_narrative)
    logging.info("Process completed and README.md generated.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
