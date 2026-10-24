# ============================================================
# HEART DISEASE DETECTION - SEMINAR 4
# Python | Pandas | Scikit-learn | Gradio | Render
# ============================================================

import os
from pathlib import Path

import pandas as pd
import gradio as gr

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ------------------------------------------------------------
# 1. DATASET
# ------------------------------------------------------------

DATASET_FILE = Path(__file__).resolve().parent / "heart.csv"

REQUIRED_COLUMNS = [
    "Age", "Sex", "Blood Pressure", "Cholesterol", "Heart Rate",
    "ECG Results", "Blood Sugar", "Chest Pain", "Target"
]

NUMERIC_COLUMNS = [
    "Age", "Blood Pressure", "Cholesterol", "Heart Rate"
]

CATEGORICAL_COLUMNS = [
    "Sex", "ECG Results", "Blood Sugar", "Chest Pain"
]

CATEGORY_MAPS = {
    "Sex": {
        "0": "Female", "0.0": "Female", "female": "Female",
        "1": "Male", "1.0": "Male", "male": "Male"
    },
    "ECG Results": {
        "0": "Normal", "0.0": "Normal", "normal": "Normal",
        "1": "Abnormal", "1.0": "Abnormal", "abnormal": "Abnormal"
    },
    "Blood Sugar": {
        "0": "Normal", "0.0": "Normal", "normal": "Normal",
        "1": "High", "1.0": "High", "high": "High"
    },
    "Chest Pain": {
        "0": "None", "0.0": "None", "none": "None",
        "1": "Mild", "1.0": "Mild", "mild": "Mild",
        "2": "Moderate", "2.0": "Moderate", "moderate": "Moderate",
        "3": "Severe", "3.0": "Severe", "severe": "Severe"
    },
    "Target": {
        "0": "No Disease", "0.0": "No Disease",
        "no disease": "No Disease", "no_disease": "No Disease",
        "1": "Disease", "1.0": "Disease", "disease": "Disease"
    }
}

ALLOWED_CATEGORIES = {
    "Sex": {"Male", "Female"},
    "ECG Results": {"Normal", "Abnormal"},
    "Blood Sugar": {"Normal", "High"},
    "Chest Pain": {"None", "Mild", "Moderate", "Severe"},
    "Target": {"Disease", "No Disease"}
}

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"heart.csv not found at {DATASET_FILE}. "
        "Keep heart.csv in the same folder as this Python file."
    )

df = pd.read_csv(DATASET_FILE)
df.columns = df.columns.astype(str).str.strip()

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

if missing:
    raise ValueError(
        "heart.csv is missing columns: " + ", ".join(missing)
    )

df = df[REQUIRED_COLUMNS].copy()

for col in NUMERIC_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col, mapping in CATEGORY_MAPS.items():
    values = df[col].astype("string").str.strip()
    normalized = values.str.lower().map(mapping)
    df[col] = normalized.fillna(values)

df = df.dropna(subset=REQUIRED_COLUMNS)
df = df.drop_duplicates().reset_index(drop=True)

if len(df) < 4 or df["Target"].nunique() < 2:
    raise ValueError(
        "Dataset needs at least 4 usable rows and two Target classes."
    )

for col, allowed in ALLOWED_CATEGORIES.items():
    unexpected = set(df[col].dropna().unique()) - allowed
    if unexpected:
        raise ValueError(
            f"Unexpected values in {col}: {unexpected}. "
            "Check the corresponding values in heart.csv."
        )

X = df.drop(columns=["Target"])
y = df["Target"].astype(str)


# ------------------------------------------------------------
# 2. MACHINE LEARNING MODEL
# ------------------------------------------------------------

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), NUMERIC_COLUMNS),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        CATEGORICAL_COLUMNS
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X, y)


# ------------------------------------------------------------
# 3. MODEL EVALUATION
# ------------------------------------------------------------

evaluation_text = "Cross-validation unavailable."
class_counts = y.value_counts()

if len(class_counts) >= 2 and int(class_counts.min()) >= 2:
    try:
        folds = min(5, int(class_counts.min()))

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

        scores = cross_val_score(
            model, X, y, cv=cv, scoring="accuracy"
        )

        evaluation_text = (
            f"Mean cross-validation accuracy: "
            f"{scores.mean() * 100:.1f}% ({folds}-fold evaluation)"
        )
    except Exception:
        evaluation_text = "Cross-validation could not be completed."


# ------------------------------------------------------------
# 4. PREDICTION
# ------------------------------------------------------------

def predict_heart_disease(
    age, sex, blood_pressure, cholesterol, heart_rate,
    ecg_results, blood_sugar, chest_pain
):
    numeric_inputs = {
        "Age": age,
        "Blood Pressure": blood_pressure,
        "Cholesterol": cholesterol,
        "Heart Rate": heart_rate
    }

    for name, value in numeric_inputs.items():
        if value is None:
            return "### Please fill in all numeric fields."

        try:
            numeric_inputs[name] = float(value)
        except (TypeError, ValueError):
            return f"### Please enter a valid number for {name}."

        if not pd.notna(numeric_inputs[name]):
            return f"### Please enter a valid number for {name}."

    if not 1 <= numeric_inputs["Age"] <= 120:
        return "### Age must be between 1 and 120."

    if not 50 <= numeric_inputs["Blood Pressure"] <= 300:
        return "### Blood pressure must be between 50 and 300."

    if not 50 <= numeric_inputs["Cholesterol"] <= 700:
        return "### Cholesterol must be between 50 and 700."

    if not 20 <= numeric_inputs["Heart Rate"] <= 300:
        return "### Heart rate must be between 20 and 300."

    categories = {
        "Sex": sex,
        "ECG Results": ecg_results,
        "Blood Sugar": blood_sugar,
        "Chest Pain": chest_pain
    }

    for name, value in categories.items():
        if value not in ALLOWED_CATEGORIES[name]:
            return f"### Please select a valid option for {name}."

    patient = pd.DataFrame([{
        **numeric_inputs,
        **categories
    }])

    prediction = str(model.predict(patient)[0])

    try:
        probabilities = model.predict_proba(patient)[0]
        class_index = list(model.classes_).index(prediction)
        model_score = probabilities[class_index] * 100
        score_line = (
            f"**Model score for this predicted class:** {model_score:.1f}%"
        )
    except Exception:
        score_line = "**Model score:** unavailable"

    if prediction == "Disease":
        result = """
## ⚠️ Model Output: Disease

The model matched the entered values to a pattern labelled
**Disease** in its training data.
"""
    else:
        result = """
## ✅ Model Output: No Disease

The model matched the entered values to a pattern labelled
**No Disease** in its training data.

This result does not rule out heart disease.
"""

    result += f"""

{score_line}

---

**Medical disclaimer:** This is an educational demonstration, not a
medical diagnosis or a clinically validated risk estimate. The model
score is not medical certainty. Consult a qualified healthcare
professional for health concerns.
"""

    return result


# ------------------------------------------------------------
# 5. RESET FORM
# ------------------------------------------------------------

def reset_form():
    return (
        45, "Male", 130, 220, 150,
        "Normal", "Normal", "Mild", ""
    )


# ------------------------------------------------------------
# 6. GRADIO INTERFACE
# ------------------------------------------------------------

custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
#hero {
    text-align: center;
    padding: 18px 8px;
}
#predict-button {
    font-weight: 700;
    min-height: 48px;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="red",
        secondary_hue="slate"
    ),
    css=custom_css,
    title="Heart Disease Detection | Minor Project"
) as demo:

    gr.Markdown("""
<div id="hero">

# ❤️ Heart Disease Detection

### Machine Learning Based Prediction System

**Minor Project | Computer Engineering | Seminar 4**

</div>
""")

    gr.Markdown("""
> **Educational prototype only.** This application is not a medical
> diagnostic tool. Predictions depend on the training data.
""")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Personal Information")

            age = gr.Number(
                label="Age (years)",
                value=45,
                minimum=1,
                maximum=120,
                precision=0
            )

            sex = gr.Dropdown(
                ["Male", "Female"],
                label="Sex",
                value="Male"
            )

            chest_pain = gr.Dropdown(
                ["None", "Mild", "Moderate", "Severe"],
                label="Chest Pain Category",
                value="Mild"
            )

        with gr.Column():
            gr.Markdown("### 🩺 Health Information")

            blood_pressure = gr.Number(
                label="Blood Pressure (dataset value)",
                value=130
            )

            cholesterol = gr.Number(
                label="Cholesterol (dataset value)",
                value=220
            )

            heart_rate = gr.Number(
                label="Heart Rate (dataset value)",
                value=150
            )

            ecg_results = gr.Dropdown(
                ["Normal", "Abnormal"],
                label="ECG Results",
                value="Normal"
            )

            blood_sugar = gr.Dropdown(
                ["Normal", "High"],
                label="Blood Sugar",
                value="Normal"
            )

    with gr.Row():
        predict_button = gr.Button(
            "🔍 Predict",
            variant="primary",
            elem_id="predict-button"
        )

        reset_button = gr.Button("↺ Reset Form")

    result = gr.Markdown()

    predict_button.click(
        fn=predict_heart_disease,
        inputs=[
            age, sex, blood_pressure, cholesterol, heart_rate,
            ecg_results, blood_sugar, chest_pain
        ],
        outputs=result
    )

    reset_button.click(
        fn=reset_form,
        inputs=[],
        outputs=[
            age, sex, blood_pressure, cholesterol, heart_rate,
            ecg_results, blood_sugar, chest_pain, result
        ]
    )

    gr.Markdown("---")
    gr.Markdown("### 📊 Model Information")

    gr.Markdown(f"""
- **Algorithm:** Logistic Regression
- **Learning type:** Supervised Machine Learning
- **Usable records:** {len(df)}
- **Evaluation:** {evaluation_text}

**Evaluation limitation:** A small dataset can produce unstable results.
Cross-validation accuracy from a tiny dataset does not establish
real-world medical accuracy.
""")

    gr.Markdown("""
### 🛠️ Technologies

Python · Pandas · Scikit-learn · Gradio

### ℹ️ About This Project

This application demonstrates data preprocessing, model training,
and a web interface for a student machine learning project.
""")


# ------------------------------------------------------------
# 7. START SERVER - RENDER
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("HEART DISEASE DETECTION - SEMINAR 4")
    print("=" * 55)
    print(f"Dataset file: {DATASET_FILE}")
    print(f"Usable dataset records: {len(df)}")
    print(evaluation_text)

    share_enabled=True
    

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=share_enabled
    )