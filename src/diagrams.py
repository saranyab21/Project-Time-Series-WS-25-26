import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load results
# ---------------------------------------------------------------------
summary_path = "C:/Users/admin/OneDrive/Desktop/SEM-2-3/Project-Time Series/gait-mamba/reports/baseline_result_summary.csv"  
df = pd.read_csv(summary_path)

dataset_order = ["Left", "Right", "Combined"]
model_order = ["RandomForest", "SVM_Linear", "SVM_RBF"]

df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
df = df.sort_values(["Dataset", "Model"])

model_labels = {
    "RandomForest": "Random Forest",
    "SVM_Linear": "SVM (Linear)",
    "SVM_RBF": "SVM (RBF)"
}

fig_dir = Path("reports/figs")
fig_dir.mkdir(parents=True, exist_ok=True)

# color_palette
color_blue = "#1f77b4"   # dark blue
color_teal = "#00a6a6"   # teal
color_grey = "#6c757d"   # neutral grey
model_colors = {
    "RandomForest": color_blue,
    "SVM_Linear": color_teal,
    "SVM_RBF": color_grey,
}

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14
})

# =========================================================
# 1) Grouped bar plot: AUC by Dataset × Model
# =========================================================
fig, ax = plt.subplots(figsize=(9, 4.8))

n_datasets = len(dataset_order)
n_models = len(model_order)
bar_width = 0.22
x = range(n_datasets)

for i, model in enumerate(model_order):
    sub = df[df["Model"] == model]
    means = sub["Mean_AUC"].values
    stds = sub["Std_AUC"].values
    x_pos = [xi + (i - (n_models-1)/2) * bar_width for xi in x]

    ax.bar(
        x_pos, means,
        width=bar_width,
        yerr=stds,
        capsize=4,
        label=model_labels[model],
        color=model_colors[model],
        edgecolor="black",
        linewidth=0.5
    )

ax.set_xticks(list(x))
ax.set_xticklabels(dataset_order)
ax.set_ylabel("Mean ROC-AUC (5-fold CV)")
ax.set_xlabel("Feature Set")
ax.set_title("Baseline Models – Cross-Validated AUC by Dataset")
ax.set_ylim(0.55, 0.95)  # zoom in so differences are visible
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.legend(title="Model", loc="lower right", frameon=True)

ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()
out_path1 = fig_dir / "baseline_auc_grouped_bar_FAU.png"
fig.savefig(out_path1, bbox_inches="tight")
print(f"Saved: {out_path1}")

plt.close(fig)

# =========================================================
# 2) Combined-only plot: Best baseline models
# =========================================================
combined = df[df["Dataset"] == "Combined"].copy()
combined["ModelLabel"] = combined["Model"].map(model_labels)

fig, ax = plt.subplots(figsize=(7.5, 4.8))

x2 = range(len(combined))
means2 = combined["Mean_AUC"].values
stds2 = combined["Std_AUC"].values
colors2 = [model_colors[m] for m in combined["Model"]]

bars = ax.bar(
    x2, means2,
    yerr=stds2,
    capsize=5,
    color=colors2,
    edgecolor="black",
    linewidth=0.5
)

ax.set_xticks(list(x2))
ax.set_xticklabels(combined["ModelLabel"], rotation=15, ha="right")
ax.set_ylabel("Mean ROC-AUC (5-fold CV)")
ax.set_title("Combined Features – Best Baseline Models")

# Focused y-range so bars fill the plot nicely
ax.set_ylim(0.75, 0.9)
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.tick_params(axis="y", labelsize=12)
ax.tick_params(axis="x", labelsize=12)

# --- annotating values clearly above bars ---
for xi, yi in zip(x2, means2):
    ax.text(
        xi, yi + 0.004,           # small offset above bar
        f"{yi:.2f}",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black"
    )

fig.tight_layout()
out_path2 = fig_dir / "baseline_auc_combined_bar_FAU.png"
fig.savefig(out_path2, bbox_inches="tight")
print(f"Saved: {out_path2}")

plt.close(fig)

