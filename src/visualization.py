"""
visualization.py
----------------
Creates and saves Matplotlib (PNG) and Plotly (HTML) visualizations.

Saves:
  outputs/plots/plot_1_exam_score_by_mode.png   — bar chart (Matplotlib)
  outputs/plots/plot_2_study_hours_hist.png      — histogram (Matplotlib)
  outputs/plots/plot_3_focus_vs_exam.png         — scatter (Matplotlib)
  outputs/plots/plot_4_interactive_scatter.html  — Plotly scatter
  outputs/plots/plot_5_interactive_bar.html      — Plotly bar
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plotly is used for interactive HTML plots.
# If not installed, plots are generated via vanilla HTML + Plotly CDN (no local install needed).
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_visualizations(filepath: str) -> None:
    os.makedirs("outputs/plots", exist_ok=True)
    df = pd.read_csv(filepath)

    # ------------------------------------------------------------------ #
    # MATPLOTLIB PLOT 1: Bar chart — Average Exam Score by Learning Mode
    # ------------------------------------------------------------------ #
    avg_score = df.groupby("Learning_Mode")["Exam_Score"].mean()

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#FF9800"]
    ax1.bar(avg_score.index, avg_score.values, color=colors, edgecolor="black", width=0.5)
    ax1.set_title("Average Exam Score by Learning Mode", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Learning Mode")
    ax1.set_ylabel("Average Exam Score")
    ax1.set_ylim(0, 100)
    for i, (mode, val) in enumerate(avg_score.items()):
        ax1.text(i, val + 1, f"{val:.1f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_1_exam_score_by_mode.png", dpi=150)
    plt.close()
    print("Saved: plot_1_exam_score_by_mode.png")

    # ------------------------------------------------------------------ #
    # MATPLOTLIB PLOT 2: Histogram — Distribution of Study Hours
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(df["Study_Hours"], bins=20, color="#4CAF50", edgecolor="black", alpha=0.85)
    ax2.set_title("Distribution of Daily Study Hours", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Study Hours")
    ax2.set_ylabel("Number of Students")
    # numpy: add mean line
    mean_hours = np.mean(df["Study_Hours"].values)
    ax2.axvline(mean_hours, color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {mean_hours:.2f} hrs")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_2_study_hours_hist.png", dpi=150)
    plt.close()
    print("Saved: plot_2_study_hours_hist.png")

    # ------------------------------------------------------------------ #
    # MATPLOTLIB PLOT 3: Scatter — Focus Level vs Exam Score (by Mode)
    # ------------------------------------------------------------------ #
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for mode, color in zip(["Online", "Offline"], ["#E91E63", "#3F51B5"]):
        subset = df[df["Learning_Mode"] == mode]
        ax3.scatter(subset["Focus_Level"], subset["Exam_Score"],
                    label=mode, alpha=0.5, color=color, s=30)
    ax3.set_title("Focus Level vs Exam Score by Learning Mode", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Focus Level (0-100)")
    ax3.set_ylabel("Exam Score (0-100)")
    ax3.legend(title="Learning Mode")
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_3_focus_vs_exam.png", dpi=150)
    plt.close()
    print("Saved: plot_3_focus_vs_exam.png")

    # ------------------------------------------------------------------ #
    # PLOTLY PLOT 4: Interactive Scatter — Study Hours vs Exam Score
    # Uses plotly via CDN so no local install needed for the HTML file
    # ------------------------------------------------------------------ #
    online  = df[df["Learning_Mode"] == "Online"]
    offline = df[df["Learning_Mode"] == "Offline"]

    scatter_traces = json.dumps([
        {
            "x": online["Study_Hours"].tolist(),
            "y": online["Exam_Score"].tolist(),
            "mode": "markers",
            "name": "Online",
            "text": ("Subject: " + online["Subject"] +
                     "<br>Focus: " + online["Focus_Level"].astype(str) +
                     "<br>Retention: " + online["Retention_Score"].astype(str)).tolist(),
            "hoverinfo": "text+x+y",
            "marker": {"color": "#E91E63", "opacity": 0.6, "size": 7}
        },
        {
            "x": offline["Study_Hours"].tolist(),
            "y": offline["Exam_Score"].tolist(),
            "mode": "markers",
            "name": "Offline",
            "text": ("Subject: " + offline["Subject"] +
                     "<br>Focus: " + offline["Focus_Level"].astype(str) +
                     "<br>Retention: " + offline["Retention_Score"].astype(str)).tolist(),
            "hoverinfo": "text+x+y",
            "marker": {"color": "#3F51B5", "opacity": 0.6, "size": 7}
        }
    ])

    html4 = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Study Hours vs Exam Score</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head><body>
<div id="chart" style="width:100%;height:600px;"></div>
<script>
var traces = {scatter_traces};
var layout = {{
  title: "Study Hours vs Exam Score (Interactive)",
  xaxis: {{title: "Study Hours"}},
  yaxis: {{title: "Exam Score"}},
  legend: {{title: {{text: "Learning Mode"}}}}
}};
Plotly.newPlot("chart", traces, layout);
</script>
</body></html>"""

    with open("outputs/plots/plot_4_interactive_scatter.html", "w") as f:
        f.write(html4)
    print("Saved: plot_4_interactive_scatter.html")

    # ------------------------------------------------------------------ #
    # PLOTLY PLOT 5: Interactive Bar — Avg Retention Score by Subject & Mode
    # ------------------------------------------------------------------ #
    grouped = df.groupby(["Subject", "Learning_Mode"])["Retention_Score"].mean().reset_index()
    subjects = grouped["Subject"].unique().tolist()

    bar_traces = []
    for mode, color in [("Online", "#E91E63"), ("Offline", "#3F51B5")]:
        sub = grouped[grouped["Learning_Mode"] == mode]
        bar_traces.append({
            "type": "bar",
            "name": mode,
            "x": sub["Subject"].tolist(),
            "y": [round(v, 2) for v in sub["Retention_Score"].tolist()],
            "marker": {"color": color}
        })

    bar_traces_json = json.dumps(bar_traces)

    html5 = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Retention Score by Subject</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head><body>
<div id="chart" style="width:100%;height:600px;"></div>
<script>
var traces = {bar_traces_json};
var layout = {{
  title: "Average Memory Retention Score by Subject and Learning Mode",
  xaxis: {{title: "Subject"}},
  yaxis: {{title: "Avg Retention Score"}},
  barmode: "group",
  legend: {{title: {{text: "Learning Mode"}}}}
}};
Plotly.newPlot("chart", traces, layout);
</script>
</body></html>"""

    with open("outputs/plots/plot_5_interactive_bar.html", "w") as f:
        f.write(html5)
    print("Saved: plot_5_interactive_bar.html")


if __name__ == "__main__":
    create_visualizations("data/dataset.csv")
