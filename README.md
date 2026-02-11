# Insurance Charges Prediction App 💰📊 (Streamlit + Machine Learning)

This project is a simple Machine Learning application built during my **45-day AI/ML internship**.  
It predicts **medical insurance charges** based on user inputs such as age, sex, BMI, children, smoker status, and region.  
The model is trained using **Linear Regression** and the interactive web application is built using **Streamlit**.

---

## 🚀 Features
- Predict insurance charges instantly using a trained ML model
- Clean and interactive Streamlit UI
- User input fields:
  - Age
  - Sex
  - BMI
  - Children
  - Smoker
  - Region
- Optional visualization:
  - Actual vs Predicted Charges graph

---

## 🧠 Machine Learning Model
- Algorithm: **Linear Regression**
- Preprocessing:
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
- Dataset: `insurance.csv`
- Train-Test Split for model evaluation

---

## 🛠️ Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 📂 Project Files Structure (GitHub)
All project files are inside the **Mitesh** folder:

```txt
Insurance-Charges-Prediction---ML/
│
└── Mitesh/
    ├── user_interface.py
    ├── insurance.csv
    ├── insurance.py
    ├── Presentation_Mitesh.pptx
    ├── Report_Mitesh.pdf
    ├── Scan_Report.pdf
    └── README.md


---

## ▶️ How to Run This Project (Step-by-Step)

### Step 1: Download / Clone Repository
```bash
git clone https://github.com/MiteshRana11/Insurance-Charges-Prediction---ML.git

### Step 2: Open Project Folder
```bash
Go inside the Mitesh folder because the main Streamlit file is there:
cd Insurance-Charges-Prediction---ML/Mitesh

### Step 3: Install Required Libraries
```bash
pip install streamlit pandas numpy matplotlib scikit-learn

### Step 4: Run the Streamlit App
```bash
streamlit run user_interface.py

### Step 5: Open in Browser
```bash
Streamlit will open automatically, otherwise open:
http://localhost:8501

---

## 📌 Output
Displays predicted insurance charges in dollars
Optional graph: Actual vs Predicted Charges

---

## 👨‍💻 Author
Mitesh Rana

---
