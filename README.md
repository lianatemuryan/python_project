# Online vs Offline Learning — ML Classification Project

## Dataset
**File:** `data/dataset.csv`  
**Source:** Provided dataset for the Python Machine Learning Mini Project.  
**Size:** 1000 rows, 6 columns

### Column Descriptions
| Column | Description |
|---|---|
| `Learning_Mode` | How the student learns — **Online** or **Offline** (TARGET) |
| `Subject` | Subject studied — English, Math, Science, Programming, History |
| `Study_Hours` | Daily study hours (1.0 – 8.0) |
| `Retention_Score` | Memory retention score (0–100) |
| `Focus_Level` | Self-reported focus level (0–100) |
| `Exam_Score` | Final exam score (0–100) |

---

## Machine Learning Task
**Type:** Binary Classification  
**Target column:** `Learning_Mode` (Online = 1, Offline = 0)  
**Model used:** Decision Tree Classifier (max_depth=5)

---

## Project Structure
```
python_project/
├── data/
│   └── dataset.csv
├── src/
│   ├── data_exploration.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── model.py
│   ├── evaluation.py
│   └── main.py
├── outputs/
│   ├── plots/
│   │   ├── plot_1_exam_score_by_mode.png
│   │   ├── plot_2_study_hours_hist.png
│   │   ├── plot_3_focus_vs_exam.png
│   │   ├── plot_4_interactive_scatter.html
│   │   └── plot_5_interactive_bar.html
│   └── results/
│       ├── metrics.txt
│       └── predictions.csv
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python src/main.py
```

---

## Results Summary
- **Accuracy:** 52.5%
- **Precision:** 60%
- **Recall:** 15%

The model struggled to separate Online from Offline students because the dataset features (study hours, exam scores, focus level) are distributed very similarly across both groups — there is no strong signal that separates the two classes. This is itself an interesting finding: learning mode alone does not strongly predict student performance metrics in this dataset.

**Possible improvements:**
- Collect additional features (e.g., internet access, device type, location)
- Try Random Forest or Logistic Regression
- Gather more data with stronger class separation
