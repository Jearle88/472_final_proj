import matplotlib.pyplot as plt

# Model labels
models = [
    "NN (All, Deep)", "NN (All, Shallow)",
    "NN (6, Deep)", "NN (6, Shallow)",
    "DT (All, Untuned)", "DT (6, No Prune)",
    "DT (6, Tuned)", "KNN (k=1)",
    "KNN (k=5)", "KNN (k=9)", "KNN (k=9, Weighted)"
]

# Corresponding accuracies (%)
accuracies = [
    99.7, 81.5,
    72.7, 55.5,
    64.6, 68.5,
    71.1, 36.4,
    63.5, 63.6, 63.6
]

# Set bar colors based on model type
colors = [
    "skyblue", "skyblue", "skyblue", "skyblue",  # Neural Nets
    "orange", "orange", "orange",                # Decision Trees
    "green", "green", "green", "green"           # KNN
]

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracies, color=colors)
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45, ha="right")
plt.title("Model Accuracy Comparison")

# Annotate each bar with its accuracy
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{acc:.1f}%", ha='center')

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
