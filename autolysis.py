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



