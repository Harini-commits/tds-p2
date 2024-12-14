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


# Set your OpenAI API key (ensure this is handled securely in production)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function: Load Dataset
def load_dataset(file_path, chunksize=None):
    """Loads the dataset with optional chunking for large files."""
    try:
        if chunksize:
            return pd.read_csv(file_path, chunksize=chunksize)
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Function: Analyze Data
def analyze_data(df):
    """Performs basic data analysis and returns summary statistics."""
    summary = {
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Missing Values": df.isnull().sum().to_dict(),
        "Data Types": df.dtypes.to_dict(),
        "Summary Statistics": df.describe(include='all').to_dict(),
    }
    return summary

# Function: Generate Visualizations
def generate_visualizations(df, output_dir="media"):
    """Generates and saves visualizations based on the dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation = df.corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    # Pairplot for numeric features (if applicable)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        sns.pairplot(df[numeric_cols])
        plt.savefig(f"{output_dir}/pairplot.png")
        plt.close()

# Function: Generate LLM Insights
def generate_llm_insights(summary, prompt_type="dataset_summary"):
    """Generates insights or narratives using OpenAI GPT."""
    try:
        prompt_map = {
            "dataset_summary": f"""
                I have a dataset with the following properties:
                - Shape: {summary['Shape']}
                - Columns: {summary['Columns']}
                - Missing Values: {summary['Missing Values']}
                - Summary Statistics: {summary['Summary Statistics']}

                Can you summarize the key insights and trends from this dataset?
            """,
            "story_generation": f"""
                Based on the following dataset properties, create a narrative:
                - {summary}
            """
        }
        prompt = prompt_map[prompt_type]

        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error generating LLM insights: {e}")
        return None

# Function: Write README
def write_readme(output_dir, summary, insights):
    """Writes a README.md file summarizing the project."""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Project 2: Automated Analysis\n")
        f.write("\n## Dataset Overview\n")
        f.write(f"- Shape: {summary['Shape']}\n")
        f.write(f"- Columns: {summary['Columns']}\n")
        f.write(f"- Missing Values: {summary['Missing Values']}\n")
        f.write("\n## Key Insights\n")
        f.write(insights or "No insights generated.")
        f.write("\n\n## Visualizations\n")
        f.write("![Correlation Heatmap](media/correlation_heatmap.png)\n")
        if os.path.exists("media/pairplot.png"):
            f.write("![Pairplot](media/pairplot.png)\n")

# Main Workflow
def main():
    # File Paths
    dataset_path = "goodreads.csv"
    output_dir = "output"

    # Step 1: Load Dataset
    df = load_dataset(dataset_path)
    if df is None:
        return

    # Step 2: Analyze Data
    summary = analyze_data(df)
    print("Dataset Summary:", summary)

    # Step 3: Generate Visualizations
    generate_visualizations(df, output_dir="media")

    # Step 4: Generate Insights with LLM
    insights = generate_llm_insights(summary)
    print("Generated Insights:", insights)

    # Step 5: Write README
    write_readme(output_dir, summary, insights)

if __name__ == "__main__":
    main()
