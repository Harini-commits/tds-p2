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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import os
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data with efficient handling for large files
def load_data(file_path):
    """Load CSV data with encoding detection and chunking for large files."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        logging.info(f"Detected encoding: {encoding}")
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)

# Generate visualizations
def create_visualizations(df, output_dir="visualizations"):
    """Generate and save visualizations to the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    visualizations = []

    try:
        # Correlation Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_path)
        plt.close()
        visualizations.append(f"Heatmap: {heatmap_path}")

        # Pair Plot (sampled for efficiency if data is too large)
        sampled_df = df.sample(min(500, len(df))) if len(df) > 500 else df
        sns.pairplot(sampled_df)
        pairplot_path = os.path.join(output_dir, "pair_plot.png")
        plt.savefig(pairplot_path)
        plt.close()
        visualizations.append(f"Pair Plot: {pairplot_path}")

    except Exception as e:
        logging.warning(f"Visualization generation failed: {e}")

    return visualizations

# Analyze dataset
def analyze_dataset(df):
    """Perform basic analysis of the dataset."""
    analysis_results = {
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Missing Values": df.isnull().sum().to_dict(),
        "Data Types": df.dtypes.to_dict(),
        "Summary Stats": df.describe().to_dict(),
    }
    return analysis_results

# Generate Markdown report
def generate_markdown_report(analysis, visualizations, output_file="report.md"):
    """Generate a Markdown report from the analysis and visualizations."""
    with open(output_file, "w") as f:
        f.write("# Dataset Analysis Report\n\n")
        f.write(f"## Dataset Overview\n")
        f.write(f"- Shape: {analysis['Shape']}\n")
        f.write(f"- Columns: {', '.join(analysis['Columns'])}\n")
        f.write(f"- Missing Values: {analysis['Missing Values']}\n\n")

        f.write("## Summary Statistics\n")
        for column, stats in analysis['Summary Stats'].items():
            f.write(f"### {column}\n")
            for stat, value in stats.items():
                f.write(f"- {stat}: {value}\n")
            f.write("\n")

        f.write("## Visualizations\n")
        for viz in visualizations:
            viz_name = viz.split(": ")[1]
            f.write(f"![{viz_name}]({viz_name})\n")

    logging.info(f"Markdown report generated at: {output_file}")

# Main script
def main(file_path):
    """Main function to execute the analysis pipeline."""
    logging.info("Starting dataset analysis pipeline.")

    # Load data
    df = load_data(file_path)

    # Analyze data
    analysis_results = analyze_dataset(df)

    # Generate visualizations
    visualizations = create_visualizations(df)

    # Generate Markdown report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"report_{timestamp}.md"
    generate_markdown_report(analysis_results, visualizations, output_file)

    logging.info("Analysis pipeline completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python script.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)




