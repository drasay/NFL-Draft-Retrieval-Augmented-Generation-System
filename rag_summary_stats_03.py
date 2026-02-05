import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

summary = pd.read_csv("rag_answer_summary.csv")

# Remove questions with Question Type = "Hallucination Test"
summary = summary[summary["Question Type"] != "Hallucination Test"]

# Average Similarity Score
avg_similarity_score_minilm = (summary["Similarity_1 (MiniLM)"].sum() + summary["Similarity_2(MiniLM)"].sum()) / (2* len(summary))
avg_similarity_score_e5 = (summary["Similarity_1 (e5)"].sum() + summary["Similarity_2 (e5)"].sum()) / (2* len(summary))

print(f"Average Similarity Score (MiniLM): {avg_similarity_score_minilm:.4f}")
print(f"Average Similarity Score (e5): {avg_similarity_score_e5:.4f}")

# Average Accuracy Score
avg_accuracy_score_minilm = summary["Accuracy (MiniLM)"].mean()
avg_accuracy_score_e5 = summary["Accuracy (e5)"].mean()
print(f"Average Accuracy Score (MiniLM): {avg_accuracy_score_minilm:.4f}")
print(f"Average Accuracy Score (e5): {avg_accuracy_score_e5:.4f}")

# Average Relevance Score
avg_relevance_score_minilm = summary["Relevance (MiniLM)"].mean()
avg_relevance_score_e5 = summary["Relevance (e5)"].mean()
print(f"Average Relevance Score (MiniLM): {avg_relevance_score_minilm:.4f}")
print(f"Average Relevance Score (e5): {avg_relevance_score_e5:.4f}")

# Average Usefulness Score
avg_usefulness_score_minilm = summary["Useful (MiniLM)"].mean()
avg_usefulness_score_e5 = summary["Useful (e5)"].mean()
print(f"Average Usefulness Score (MiniLM): {avg_usefulness_score_minilm:.4f}")
print(f"Average Usefulness Score (e5): {avg_usefulness_score_e5:.4f}")

# Correlation between Similarity and Accuracy
similarity_accuracy_corr_minilm = summary["Similarity_1 (MiniLM)"].corr(summary["Accuracy (MiniLM)"])
similarity_accuracy_corr_e5 = summary["Similarity_1 (e5)"].corr(summary["Accuracy (e5)"])
print(f"Correlation between Similarity and Accuracy (MiniLM): {similarity_accuracy_corr_minilm:.4f}")
print(f"Correlation between Similarity and Accuracy (e5): {similarity_accuracy_corr_e5:.4f}")

# Correlation between Similarity and Relevance
similarity_relevance_corr_minilm = summary["Similarity_1 (MiniLM)"].corr(summary["Relevance (MiniLM)"])
similarity_relevance_corr_e5 = summary["Similarity_1 (e5)"].corr(summary["Relevance (e5)"])
print(f"Correlation between Similarity and Relevance (MiniLM): {similarity_relevance_corr_minilm:.4f}")
print(f"Correlation between Similarity and Relevance (e5): {similarity_relevance_corr_e5:.4f}")

# Correlation between Similarity and Usefulness
similarity_usefulness_corr_minilm = summary["Similarity_1 (MiniLM)"].corr(summary["Useful (MiniLM)"])
similarity_usefulness_corr_e5 = summary["Similarity_1 (e5)"].corr(summary["Useful (e5)"])
print(f"Correlation between Similarity and Usefulness (MiniLM): {similarity_usefulness_corr_minilm:.4f}")
print(f"Correlation between Similarity and Usefulness (e5): {similarity_usefulness_corr_e5:.4f}")

# Accuracy and Relevance when Usefulness is 1
usefulness_1_minilm = summary[summary["Useful (MiniLM)"] == 1]
usefulness_1_e5 = summary[summary["Useful (e5)"] == 1]
avg_useful_minilm = usefulness_1_minilm["Accuracy (MiniLM)"].mean()
avg_useful_e5 = usefulness_1_e5["Accuracy (e5)"].mean()
print(f"Average Accuracy when Usefulness is 1 (MiniLM): {avg_useful_minilm:.4f}")
print(f"Average Accuracy when Usefulness is 1 (e5): {avg_useful_e5:.4f}")

# Accuracy and Relevance when Usefulness is 0
usefulness_0_minilm = summary[summary["Useful (MiniLM)"] == 0]
usefulness_0_e5 = summary[summary["Useful (e5)"] == 0]
avg_not_useful_minilm = usefulness_0_minilm["Accuracy (MiniLM)"].mean()
avg_not_useful_e5 = usefulness_0_e5["Accuracy (e5)"].mean()
print(f"Average Accuracy when Usefulness is 0 (MiniLM): {avg_not_useful_minilm:.4f}")
print(f"Average Accuracy when Usefulness is 0 (e5): {avg_not_useful_e5:.4f}")

# Average Accuracy, Relevance, and Usefulness for each Question Type
question_types = summary["Question Type"].unique()
for qtype in question_types:
    avg_accuracy_minilm = summary[summary["Question Type"] == qtype]["Accuracy (MiniLM)"].mean()
    avg_accuracy_e5 = summary[summary["Question Type"] == qtype]["Accuracy (e5)"].mean()
    avg_relevance_minilm = summary[summary["Question Type"] == qtype]["Relevance (MiniLM)"].mean()
    avg_relevance_e5 = summary[summary["Question Type"] == qtype]["Relevance (e5)"].mean()
    avg_usefulness_minilm = summary[summary["Question Type"] == qtype]["Useful (MiniLM)"].mean()
    avg_usefulness_e5 = summary[summary["Question Type"] == qtype]["Useful (e5)"].mean()

    print(f"\n--- Question Type: {qtype} ---")
    print(f"Average Accuracy (MiniLM): {avg_accuracy_minilm:.4f}")
    print(f"Average Accuracy (e5): {avg_accuracy_e5:.4f}")
    print(f"Average Relevance (MiniLM): {avg_relevance_minilm:.4f}")
    print(f"Average Relevance (e5): {avg_relevance_e5:.4f}")
    print(f"Average Usefulness (MiniLM): {avg_usefulness_minilm:.4f}")
    print(f"Average Usefulness (e5): {avg_usefulness_e5:.4f}")

# Visualize the results
metrics = ["Accuracy", "Relevance", "Usefulness"]
models = ["MiniLM", "e5"]
question_types = ["Knowledge", "Analysis", "Application", "Synthesis"]
bar_colors = ['#1f77b4', '#ff7f0e']  # MiniLM (blue), e5 (orange)

# Precompute values into a dictionary structure

plot_data = {}
for qtype in question_types:
    plot_data[qtype] = {
        "Accuracy": [
            summary[summary["Question Type"] == qtype]["Accuracy (MiniLM)"].mean() / 5,
            summary[summary["Question Type"] == qtype]["Accuracy (e5)"].mean() / 5
        ],
        "Relevance": [
            summary[summary["Question Type"] == qtype]["Relevance (MiniLM)"].mean() / 5,
            summary[summary["Question Type"] == qtype]["Relevance (e5)"].mean() / 5
        ],
        "Usefulness": [
            summary[summary["Question Type"] == qtype]["Useful (MiniLM)"].mean() / 5,
            summary[summary["Question Type"] == qtype]["Useful (e5)"].mean() / 5
        ]
    }

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, qtype in enumerate(question_types):
    ax = axes[idx]
    metric_values = plot_data[qtype]

    x = np.arange(len(metrics))
    width = 0.35

    # Plot MiniLM and e5 bars
    minilm_values = [metric_values[m][0] for m in metrics]
    e5_values = [metric_values[m][1] for m in metrics]

    ax.bar(x - width/2, minilm_values, width, label='MiniLM', color=bar_colors[0])
    ax.bar(x + width/2, e5_values, width, label='e5', color=bar_colors[1])

    ax.set_title(qtype)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)  
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()