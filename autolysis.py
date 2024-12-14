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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os

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
            "Unique Values": {col: df[col].nunique() for col in df.columns}
        }
        return summary
    except Exception as e:
        print(f"Error analyzing data: {e}")
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

        # Distribution Plots for Numerical Columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.savefig(f"{output_dir}/distribution_{col}.png")
            plt.close()

        # Bar Plots for Categorical Columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            df[col].value_counts().plot(kind='bar', color='skyblue')
            plt.title(f"Bar Plot of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.savefig(f"{output_dir}/barplot_{col}.png")
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
                - Unique Values: {summary.get('Unique Values', 'N/A')}
                - Summary Statistics: {summary.get('Summary Statistics', 'N/A')}

                Can you summarize the key insights, trends, and potential issues from this dataset?
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

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating LLM insights: {e}")
        return None

# Function: Write README
def write_readme(output_dir, summary, insights):
    """Writes a README.md file summarizing the project."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write("# Project 2: Automated Analysis\n")
            f.write("\n## Objective\n")
            f.write("Automate dataset analysis and generate insights using Python and GPT models.\n")

            f.write("\n## Dataset Overview\n")
            f.write(f"- Shape: {summary.get('Shape', 'N/A')}\n")
            f.write(f"- Columns: {summary.get('Columns', 'N/A')}\n")
            f.write(f"- Missing Values: {summary.get('Missing Values', 'N/A')}\n")
            f.write(f"- Unique Values: {summary.get('Unique Values', 'N/A')}\n")

            f.write("\n## Key Insights\n")
            f.write(insights or "No insights generated.")

            f.write("\n\n## Visualizations\n")
            if os.path.exists("media/correlation_heatmap.png"):
                f.write("![Correlation Heatmap](media/correlation_heatmap.png)\n")
            if os.path.exists("media/pairplot.png"):
                f.write("![Pairplot](media/pairplot.png)\n")
            for file in os.listdir("media"):
                if file.endswith(".png") and file not in ["correlation_heatmap.png", "pairplot.png"]:
                    f.write(f"![{file.split('.')[0]}](media/{file})\n")
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

    # Step 3: Generate Visualizations
    generate_visualizations(df, output_dir="media")

    # Step 4: Generate Insights with LLM
    insights = generate_llm_insights(summary)
    print("Generated Insights:", insights)

    # Step 5: Write README
    write_readme(output_dir, summary, insights)

if __name__ == "__main__":
    main()

