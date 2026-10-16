# 🎓 Student Placement Predictor

An end-to-end machine learning project on a small toy dataset: from raw data, to model training, to a deployed web app. Given a student's **CGPA** and **IQ**, the app predicts whether the student will be **placed**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://va2nttsqbrxhlp3ij4t2dd.streamlit.app/)

### 🔗 [Live Demo](https://end-to-end-toy-project-v2hm8bvnqcwffisfjpgcqc.streamlit.app/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project walks through the full ML lifecycle:

1. Load and preprocess the data (`placement.csv`)
2. Run exploratory data analysis (EDA) and select features
3. Extract inputs (`cgpa`, `iq`) and output (`placement`)
4. Split into train and test sets
5. Train a **Logistic Regression** classifier
6. Evaluate on the test set (~95% accuracy)
7. Serialize the model and deploy it as an interactive **Streamlit** app

## Features

- Interactive web UI for entering CGPA and IQ
- Binary prediction: `PLACED` / `NOT PLACED`
- Prediction confidence (class probability)
- One-command retraining via `train.py`
- Auto-redeploy on every push to `main` (Streamlit Community Cloud)

## Project Structure

```
End-to-End-Toy-Project/
├── End_to_End_Toy.ipynb   # Notebook: EDA, training, evaluation
├── app.py                 # Streamlit app (loads model, serves predictions)
├── train.py               # Standalone script to retrain the model
├── placement.csv          # Dataset (100 rows: cgpa, iq, placement)
├── requirements.txt       # Pinned Python dependencies
└── README.md
```

## Dataset

`placement.csv` contains 100 samples:

| Column      | Type  | Description                            | Range     |
|-------------|-------|----------------------------------------|-----------|
| `cgpa`      | float | Student's CGPA                         | 3.3 – 8.5 |
| `iq`        | float | Student's IQ score                     | 37 – 233  |
| `placement` | int   | Target: `1` = placed, `0` = not placed | 0 / 1     |

> **Note:** This is a small, toy dataset. Some values (e.g. IQ of 37 or 233) are outside realistic ranges, so treat the data as illustrative rather than real-world.

## Model

| Property          | Value                                                |
|-------------------|------------------------------------------------------|
| Algorithm         | Logistic Regression (`sklearn.linear_model`)         |
| Features          | `cgpa`, `iq`                                         |
| Target            | `placement`                                          |
| Test accuracy     | ~95%                                                 |
| Serialization     | `pickle`                                             |

The trained model is saved with `pickle` and loaded by the Streamlit app at startup.

## Getting Started

### Prerequisites

- Python 3.9+
- `git`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/hima879/-End-to-End-Toy-Project.git End-to-End-Toy-Project
cd End-to-End-Toy-Project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### (Optional) Retrain the model

```bash
python train.py
```

### Run the app

```bash
streamlit run app.py
```

Then open <http://localhost:8501> in your browser.

## Usage

1. Enter a student's **CGPA** and **IQ**.
2. Click **Predict Placement**.
3. The app shows:
   - the prediction (`PLACED` / `NOT PLACED`)
   - the confidence score (probability)

## Deployment

The app is hosted on **Streamlit Community Cloud**. Every push to `main` triggers an automatic redeploy. To redeploy manually, open the app dashboard and click **Reboot**.

> **⚠️ scikit-learn version:** Pickled models are version-sensitive. `requirements.txt` pins the same `scikit-learn` version used to train the model, which avoids `ModuleNotFoundError` and other pickle incompatibilities at load time. If you retrain with a different version, update the pin to match.

## Tech Stack

- **Python 3**
- **scikit-learn**: modeling
- **pandas / numpy**: data handling
- **Streamlit**: web app
- **pickle**: model serialization

## Limitations

- Only 100 samples, so the ~95% accuracy comes from a very small test set and may not generalize.
- Only two features; real placement depends on many more factors (skills, projects, internships, communication, etc.).
- The data appears synthetic, so predictions are for demonstration only and should not be used for real decisions about people.

## Future Improvements

- Add cross-validation and more metrics (precision, recall, F1, confusion matrix)
- Compare against other models (Random Forest, SVM, etc.)
- Add more features and a larger dataset
- Add input validation and a decision-boundary plot in the app
- Add unit tests and a CI workflow

## License

This is a toy/educational project, free to use however you like. Consider adding a `LICENSE` file (e.g. MIT) to make this explicit.
