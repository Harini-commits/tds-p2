# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "numpy",
#   "ipykernel",
#   "openai"
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
import openai

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function: Load Dataset
def load_dataset(file_path):
    """Loads the dataset."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        return None

# Function: Analyze Data
def analyze_data(df):
    """Performs basic data analysis and returns summary statistics."""
    return {
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Missing Values": df.isnull().sum().to_dict(),
        "Data Types": df.dtypes.to_dict(),
        "Summary Statistics": df.describe(include='all', datetime_is_numeric=True).to_dict(),
    }

# Function: Generate Visualizations
def generate_visualizations(df, output_dir="media"):
    """Generates and saves visualizations based on the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()

        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            sns.pairplot(df[numeric_cols])
            plt.savefig(f"{output_dir}/pairplot.png")
            plt.close()
    except Exception as e:
        print(f"Error generating visualizations: {e}")

# Function: Generate LLM Insights
def generate_llm_insights(summary, prompt_type="dataset_summary"):
    """Generates insights or narratives using OpenAI GPT."""
    try:
        prompt_map = {
            "dataset_summary": f"""
                I have a dataset with these properties:
                - Shape: {summary['Shape']}
                - Columns: {summary['Columns']}
                - Missing Values: {summary['Missing Values']}
                - Summary Statistics: {summary['Summary Statistics']}

                Summarize the key insights and trends.
            """,
            "trend_analysis": "Analyze trends in the dataset and highlight anomalies.",
        }
        prompt = prompt_map.get(prompt_type, "Provide a general analysis.")
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error generating LLM insights: {e}")
        return "No insights generated due to an error."

# Function: Write README
def write_readme(output_dir, summary, insights):
    """Writes a README.md file summarizing the project."""
    os.makedirs(output_dir, exist_ok=True)
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Project 2: Automated Analysis\n")
        f.write("\n## Dataset Overview\n")
        f.write(f"- Shape: {summary['Shape']}\n")
        f.write(f"- Columns: {summary['Columns']}\n")
        f.write(f"- Missing Values: {summary['Missing Values']}\n")
        f.write("\n## Key Insights\n")
        f.write(insights or "No insights available.\n")
        f.write("\n## Visualizations\n")
        f.write("- Correlation Heatmap: `media/correlation_heatmap.png`\n")
        if os.path.exists("media/pairplot.png"):
            f.write("- Pairplot: `media/pairplot.png`\n")

# Main Workflow
def main():
    datasets = ["goodreads.csv", "happiness.csv", "media.csv"]
    output_dir = "output"

    for dataset_path in datasets:
        print(f"Processing {dataset_path}...")

        df = load_dataset(dataset_path)
        if df is None:
            continue

        summary = analyze_data(df)
        print(f"Summary for {dataset_path}: {summary}")

        generate_visualizations(df, output_dir="media")
        insights = generate_llm_insights(summary)
        print(f"Insights for {dataset_path}: {insights}")

        write_readme(output_dir, summary, insights)

if __name__ == "__main__":
    main()
