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
#   "openai",
#   "scikit-learn"
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind

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
    try:
        summary = {
            "Shape": df.shape,
            "Columns": df.columns.tolist(),
            "Missing Values": df.isnull().sum().to_dict(),
            "Data Types": df.dtypes.to_dict(),
            "Summary Statistics": df.describe(include='all').to_dict(),
        }

        # Additional Analysis: Outliers and Distribution
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            summary[f"Outliers in {col}"] = {
                "Q1": df[col].quantile(0.25),
                "Q3": df[col].quantile(0.75),
                "IQR": df[col].quantile(0.75) - df[col].quantile(0.25),
                "Outliers": ((df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | \
                             (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))).sum()
            }

        # Additional Analysis: Correlation with Target Variable (if applicable)
        if "target" in df.columns:
            correlations = df.corr(numeric_only=True)["target"].sort_values(ascending=False).to_dict()
            summary["Target Correlations"] = correlations

        return summary
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return {}

# Function: Advanced Statistical Analysis
def perform_statistical_analysis(df):
    """Performs advanced statistical tests and returns insights."""
    try:
        insights = {}

        # Example: Hypothesis Testing
        if "target" in df.columns and df["target"].dtype in ['float64', 'int64']:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if col != "target":
                    stat, p_value = ttest_ind(df["target"], df[col], nan_policy='omit')
                    insights[f"T-test between target and {col}"] = {
                        "Statistic": stat,
                        "P-value": p_value
                    }

        # Example: Clustering
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1)
        if len(numeric_cols.columns) > 1:
            kmeans = KMeans(n_clusters=3, random_state=42).fit(numeric_cols)
            insights["Clustering Labels"] = kmeans.labels_.tolist()

        return insights
    except Exception as e:
        print(f"Error in statistical analysis: {e}")
        return {}

# Function: Generate Visualizations
def generate_visualizations(df, output_dir="media"):
    """Generates and saves visualizations based on the dataset."""
    try:
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

        # Distribution Plots
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), kde=True, color="blue")
            plt.title(f"Distribution of {col}")
            plt.savefig(f"{output_dir}/{col}_distribution.png")
            plt.close()
    except Exception as e:
        print(f"Error generating visualizations: {e}")

# Function: Generate LLM Insights
def generate_llm_insights(summary, prompt_type="dataset_summary"):
    """Generates insights or narratives using OpenAI GPT."""
    try:
        prompt_map = {
            "dataset_summary": f"""
                I have a dataset with the following properties:
                - Shape: {summary.get('Shape', 'N/A')}
                - Columns: {summary.get('Columns', 'N/A')}
                - Missing Values: {summary.get('Missing Values', 'N/A')}
                - Summary Statistics: {summary.get('Summary Statistics', 'N/A')}

                Can you summarize the key insights and trends from this dataset?
            """,
            "story_generation": f"""
                Based on the following dataset properties, create a narrative:
                - {summary}
            """
        }
        prompt = prompt_map.get(prompt_type, "")

        if not prompt:
            print("Invalid prompt type")
            return None

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
    """Writes a README file with dataset summary and generated insights."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write("# Dataset Insights\n\n")
            f.write("## Summary\n")
            f.write(f"Shape: {summary.get('Shape', 'N/A')}\n\n")
            f.write("## Columns\n")
            f.write(f"{summary.get('Columns', 'N/A')}\n\n")
            f.write("## Missing Values\n")
            f.write(f"{summary.get('Missing Values', 'N/A')}\n\n")
            f.write("## Insights\n")
            f.write(f"{insights}\n")
        print(f"README generated at {readme_path}")
    except Exception as e:
        print(f"Error writing README: {e}")

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

    # Step 3: Perform Statistical Analysis
    stats_insights = perform_statistical_analysis(df)
    print("Statistical Insights:", stats_insights)

    # Step 4: Generate Visualizations
    generate_visualizations(df, output_dir="media")

    # Step 5: Generate Insights with LLM
    insights = generate_llm_insights(summary)
    print("Generated Insights:", insights)

    # Step 6: Write README
    write_readme(output_dir, summary, insights)

if __name__ == "__main__":
    main()



