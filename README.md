
## Student Mental Health Prediction App

**Calm Health** is a machine learning–powered web application designed to predict the **depression risk level** of students based on academic, social, and personal factors.
Built with **Streamlit** and a **Random Forest Classifier**, this project helps identify students who may be at a higher risk of mental health challenges, promoting early awareness and intervention.

---

###  Project Overview

University students face numerous pressures — academic workload, financial stress, social isolation, and uncertainty about the future. These factors can contribute to declining mental well-being.
**Calm Health** aims to provide an early indication of a student’s potential depression risk level using AI models trained on real-world data.

---

###  Features

*  **Depression Risk Prediction** (Low / Moderate / High)
*  **Probability Distribution Visualization**
*  **“Calm Health” Theme Design** — soothing color palette and minimal interface
*  **Context-aware feedback** summarizing possible factors behind each prediction
*  **Multi-course flexibility** — model works across various academic programs
*  Built with open-source tools: `Streamlit`, `Scikit-learn`, `Pandas`, and `Joblib`

---

###  Tech Stack

| Component                  | Technology Used          |
| -------------------------- | ------------------------ |
| **Programming Language**   | Python                   |
| **Web Framework**          | Streamlit                |
| **Machine Learning Model** | Random Forest Classifier |
| **Data Handling**          | Pandas, NumPy            |
| **Model Serialization**    | Joblib                   |
| **Visualization**          | Streamlit Charts         |
| **Version Control**        | Git & GitHub             |

---

###  Installation & Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/Iamkvng01/Mental-Health-Prediction.git
   cd Mental-Health-Prediction
   ```

2. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # on macOS/Linux
   venv\Scripts\activate      # on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are available**

   * `mental_health_rf_model.pkl`
   * `feature_columns.pkl`
   * `feature_dtypes.pkl`

5. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

   or if your main file is named `app.py`:

   ```bash
   streamlit run app.py
   ```

6. **View it in your browser**

   ```
   http://localhost:8501
   ```

---

### Model Details

* **Algorithm:** Random Forest Classifier

* **Input Features:**

  * Gender, Age, Year of Study, CGPA
  * Academic Workload, Academic Pressure, Financial Concerns
  * Social Relationships, Average Sleep, Study Satisfaction
  * Anxiety, Isolation, Future Insecurity, Panic Attacks, Risk Level
  * Course of Study (encoded)

* **Output Classes:**

  * Low
  * Moderate
  * High

---

###  Project Structure

```
Mental-Health-Prediction/
│
├── app.py                      # Streamlit frontend
├── mental_health_rf_model.pkl   # Trained Random Forest model
├── feature_columns.pkl          # Model feature list
├── feature_dtypes.pkl           # Feature data types
├── requirements.txt             # Dependencies
├── .streamlit/
│   └── config.toml              # UI theme configuration
└── README.md                    # Project documentation
```

---

### Example Output

After entering student details, the app predicts:

> **Predicted Depression Level:** Moderate
>
> *Based on your sleep hours, isolation level, and academic pressure, your depression risk level is moderate.*
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/8a637ac5-9b6f-4368-b84d-823126f2562a" />


---

###  Author

**Developed by:** [Godenaan](https://github.com/Iamkvng01)

---

###  Disclaimer

This application is intended **for educational and research purposes only**.
