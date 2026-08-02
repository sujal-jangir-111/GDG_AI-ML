# 💊 Medicine Price & Generic Alternative Finder

An AI-powered web application that estimates a **fair market price** for prescription medicines in India and automatically discovers **cheaper generic alternatives** with the exact same active composition.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)

---

## 🔗 Live Demo & Deployment

* **Live Web App:** `[Insert Your Deployed Streamlit / Render Link Here]`
* **FastAPI Backend Documentation:** `[Insert Your Deployed API Link Here]/docs`

---

## 📌 Problem Statement & Solution

In India, the same medicine (with the exact same chemical composition and salt) is often sold by different pharmaceutical companies at wildly different prices. For example:
* **Brand A** sells Paracetamol 500mg for **₹15**
* **Brand B** sells Paracetamol 500mg for **₹60**

Most consumers do not know this and end up paying significantly more than necessary. 

**This app solves the problem in two ways:**
1. **AI Fair Price Predictor:** Predicts an expected "Fair Market Price" for a medicine based on its active ingredients, dosage form, pack size, and manufacturer pricing history.
2. **Cheaper Generic Alternative Finder:** Searches the dataset for alternative brands with the **exact same chemical composition** that cost less money per unit (price per tablet/ml).

---

## 🚀 Key Features

* 🤖 **AI Fair Market Price Prediction:** Calculates a fair market price estimate in plain Indian Rupees (₹).
* 📉 **Cheaper Alternatives Search:** Displays a list of top 3–5 cheaper generic substitutes matching the exact composition.
* 🏢 **Manufacturer Price Comparison Chart:** Renders an interactive bar chart comparing prices across different pharmaceutical manufacturers for that composition.
* 📷 **OCR Medicine Strip Scanner (Optional Feature):** Upload a photo of a medicine strip or box, and the app reads the medicine name automatically using `EasyOCR` and fuzzy text matching (`difflib`).
* ⚡ **FastAPI + Streamlit Architecture:** Decoupled production-grade REST API backend with an interactive web user interface.

---

## 🧹 Data Preprocessing & Cleaning Approach

Working with raw Indian pharmaceutical data (~250,000 rows) requires careful text standardization and anomaly filtering:

### 1. Text Standardization (`cleaned_composition`)
Composition text in raw datasets is notoriously messy (e.g., `"Paracetamol (500mg)"`, `"PARACETAMOL-500"`, `"paracetamol / 500 mg"`).
* All composition text was lowercased.
* Parentheses, brackets, hyphens, and inconsistent spaces were stripped.
* Dosage text and forward slashes (`/`) were standardized.
* Result: Produced a clean, unified `cleaned_composition` column that allows reliable matching across different manufacturer brand names.

### 2. Anomaly Removal & Outlier Capping
* **`price_inr > 0`:** Removed invalid zero-priced records (missing data or non-commercial public distribution vaccine schemes).
* **`price_inr <= 3000`:** Applied 99th percentile upper-tail outlier capping.
  * *Reasoning:* Exploratory Data Analysis (EDA) revealed that 99% of consumer medicines cost under **₹2,899** (Median = ₹79, 75th Percentile = ₹140). Extreme 0.01% outliers (up to ₹4,36,000 for rare imported cancer biologics) distorted decision tree split thresholds. Capping at ₹3,000 drastically improved overall model accuracy (MAE) for 99% of everyday consumer drugs.

### 3. Unit Price Normalization (`unit_price`)
To ensure fair comparisons between different pack sizes (e.g., comparing a 10-tablet strip vs. a 100-tablet box):
$$\text{Unit Price} = \frac{\text{price\_inr}}{\text{pack\_size}}$$
The alternative finder sorts and recommends substitutes based on `unit_price`, preventing misleading comparisons between different box sizes.

---

## 📊 Machine Learning Model Selection & Results

We evaluated three tree-based regression models using **Mean Absolute Error (MAE)** in actual Rupees on a test split of ~50,000 medicines:

| Model | MAE (Rupees) | Model Size | Selected as Final Model? |
| :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | ₹49.81 | ~250 MB | No |
| **LightGBM Regressor** | ₹48.77 | ~15 MB | No |
| 🏆 **HistGradientBoosting Regressor** | **₹48.07** | **~8 MB** | **YES (Winner)** |

### Why HistGradientBoosting Was Selected:
1. **Lowest Error:** Achieved the lowest Mean Absolute Error (**₹48.07**), improving average prediction accuracy by ~₹1.74 per medicine compared to Random Forest.
2. **Sequential Error Correction:** Boosting builds trees sequentially to fix residual errors from previous trees, capturing complex interactions between `manufacturer` and `dosage_form`.
3. **Deployment Efficiency:** The model file is tiny (~8 MB) and loads significantly faster during cloud deployment on platforms like Render or Hugging Face Spaces.

---

## ⚙️ How Output Prediction Works

```text
[User Input: Medicine Brand Name]
               │
               ▼
┌──────────────────────────────┐
│  FastAPI Backend Lookup      │ ──► Retrieves Composition, Dosage, Pack Size, Manufacturer
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  ColumnTransformer           │ ──► Target Encodes Manufacturer & Cleaned Composition
│  (preprocessor.joblib)       │     One-Hot Encodes Dosage Form & Pack Unit
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  HistGradientBoosting Model  │ ──► Predicts Price in Transformed Scale
│  (rf_model.joblib)           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  PowerTransformer            │ ──► Inverse transforms back to actual Indian Rupees (₹)
│  (power_transformer.joblib)  │
└──────────────┬───────────────┘
               │
               ▼
[Predicted Fair Market Price (₹) & Cheaper Alternatives List]















## 💻 How to Run the Project (Step-by-Step Guide)

Follow these simple steps to set up and run both the **FastAPI Backend** and **Streamlit Frontend** on your local machine.

### Prerequisites
Make sure you have installed on your system:
* **Python 3.10** or higher
* **Git**

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
Step 2: Create and Activate a Virtual Environment
On Windows (Command Prompt / PowerShell):
code
Cmd
python -m venv venv
venv\Scripts\activate
On Mac / Linux:
code
Bash
python3 -m venv venv
source venv/bin/activate
(You should see (venv) appear at the beginning of your terminal prompt).
Step 3: Install Dependencies
Install all required libraries for both backend and frontend:
code
Bash
pip install -r backend/requirements.txt
pip install easyocr pillow numpy
Step 4: Run the FastAPI Backend Server
Open your terminal and navigate into the backend/ directory:
code
Bash
cd backend
uvicorn main:app --reload
Backend URL: http://127.0.0.1:8000
Interactive API Docs (Swagger UI): http://127.0.0.1:8000/docs
(Leave this terminal running! The backend must stay active to handle search requests).
Step 5: Run the Streamlit Frontend Website
Open a NEW terminal window, activate your virtual environment again (venv\Scripts\activate), and navigate into the frontend/ directory:
code
Bash
cd frontend
streamlit run app.py
