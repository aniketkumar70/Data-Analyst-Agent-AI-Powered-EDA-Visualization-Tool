# 🤖 Data Analyst Agent

This project implements a **Data Analyst Agent** using Python, designed to automate key tasks in data analysis workflows. It can ingest datasets, analyze and visualize data, and assist with exploratory data analysis (EDA), providing a foundation for more advanced data science or AI-assisted analytics.

> ✅ This tool is implemented in **two formats**:
> 1. A **Jupyter Notebook** with an integrated **Gradio interface**
> 2. A **Streamlit Web App** using two Python files (`streamlit_app.py`, `data_analyst_agent.py`)

---

## 🚀 Features

- 📥 **Automatic Data Ingestion**  
  Upload and parse datasets from `.csv`, `.xlsx`, or `.json` formats.

- 📊 **Exploratory Data Analysis (EDA)**  
  Summary statistics, missing value reports, distributions, and pairwise correlations.

- 📈 **Data Visualization**  
  Auto-generated plots (histograms, bar charts, heatmaps) using `matplotlib` and `seaborn`.

- 🧠 **Agent-like Functionality**  
  Logic to interpret instructions or perform actions semi-autonomously based on analysis context.

- 🌐 **Multiple Interfaces**  
  Use as an interactive Jupyter notebook (with Gradio) or a web app (Streamlit).

- 📦 **Modular & Extensible Design**  
  Well-structured code sections make it easy to add feature engineering, modeling, or dashboarding.

---

## 🧰 Tech Stack

| Component     | Description                  |
|---------------|------------------------------|
| `Python`      | Core programming language    |
| `Pandas`      | Data manipulation            |
| `NumPy`       | Numerical operations         |
| `Matplotlib`  | Basic plotting               |
| `Seaborn`     | Statistical plotting         |
| `Scikit-learn`| Optional: for preprocessing, clustering |
| `Gradio`      | UI interface in the notebook |
| `Streamlit`   | Web-based app framework      |
| `IPython` / `Notebook` | Interactive execution |

---

## 📂 Project Structure

```bash
Data_Analyst_Agent/
├── Data_Analyst_Agent.ipynb    # Jupyter Notebook with Gradio integration
├── streamlit_app.py            # Streamlit web interface
├── agent.py                    # Core agent logic for Streamlit app
├── data/                       # (Optional) Folder for sample data
├── outputs/                    # (Optional) Auto-generated plots or reports
└── README.md                   # Project documentation
