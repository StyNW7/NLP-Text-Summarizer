import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Model": ["t5-small", "facebook/bart-base", "google/pegasus-xsum", "allenai/led-base-16384"],
    "ROUGE-1": [0.1369, 0.1236, 0.1018, 0.1643],
    "ROUGE-2": [0.0359, 0.0340, 0.0463, 0.0685],
    "ROUGE-L": [0.1013, 0.1163, 0.0922, 0.1522]
}

df = pd.DataFrame(data)

df_plot = df.set_index("Model")

plt.figure(figsize=(10, 6))
df_plot.plot(kind="bar", ax=plt.gca(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.title("ROUGE Scores Comparison Across Models", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.legend(title="Metric")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig("../results/rouge_comparison.png")
plt.show()