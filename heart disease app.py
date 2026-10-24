# ============================================================
# HEART DISEASE DETECTION - SEMINAR 4 VERSION
# Python | Pandas | Scikit-learn | Gradio
# ============================================================

import os
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

DATASET_FILE = "heart.csv"

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

if os.path.exists(DATASET_FILE):
    df = pd.read_csv(DATASET_FILE)
else:
    # Educational demo data only; not real clinical data.
    df = pd.DataFrame({
        "Age": [45, 52, 38, 61, 49, 35, 58, 42, 55, 47],
        "Sex": [
            "Male", "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female", "Male"
        ],
        "Blood Pressure": [130, 145, 118, 155, 135, 120, 150, 125, 140, 128],
        "Cholesterol": [220, 245, 190, 270, 230, 185, 255, 205, 240, 215],
        "Heart Rate": [150, 120, 165, 110, 140, 175, 115, 155, 125, 145],
        "ECG Results": [
            "Normal", "Abnormal", "Normal", "Abnormal", "Normal",
            "Normal", "Abnormal", "Normal", "Abnormal", "Normal"
        ],
        "Blood Sugar": [
            "Normal", "High", "Normal", "High", "Normal",
            "Normal", "High", "Normal", "High", "Normal"
        ],
        "Chest Pain": [
            "Mild", "Severe", "None", "Severe", "Moderate",
            "None", "Severe", "Mild", "Moderate", "Mild"
        ],
        "Target": [
            "No Disease", "Disease", "No Disease", "Disease",
            "Disease", "No Disease", "Disease", "No Disease",
            "Disease", "No Disease"
        ]
    })

df.columns = df.columns.str.strip()

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(
        "heart.csv is missing these columns: " + ", ".join(missing)
    )

# Clean columns used by the model.
df = df[REQUIRED_COLUMNS].copy()

for col in NUMERIC_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in CATEGORICAL_COLUMNS + ["Target"]:
    df[col] = df[col].astype("string").str.strip()

df = df.dropna(subset=REQUIRED_COLUMNS)
df = df.drop_duplicates().reset_index(drop=True)

if len(df) < 4 or df["Target"].nunique() < 2:
    raise ValueError(
        "The dataset needs more usable rows and at least two target classes."
    )

# Ensure the categories match the interface options.
ALLOWED_CATEGORIES = {
    "Sex": {"Male", "Female"},
    "ECG Results": {"Normal", "Abnormal"},
    "Blood Sugar": {"Normal", "High"},
    "Chest Pain": {"None", "Mild", "Moderate", "Severe"},
}

for column, allowed in ALLOWED_CATEGORIES.items():
    unexpected = set(df[column].dropna().unique()) - allowed
    if unexpected:
        raise ValueError(
            f"Unexpected value(s) in {column}: {unexpected}. "
            "Update the dataset or the dropdown choices to match."
        )

X = df.drop(columns=["Target"])
y = df["Target"].astype(str)

# ------------------------------------------------------------
# 2. MACHINE LEARNING PIPELINE
# ------------------------------------------------------------

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), NUMERIC_COLUMNS),
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     CATEGORICAL_COLUMNS)
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
            f"Mean cross-validation accuracy: {scores.mean() * 100:.1f}% "
            f"({folds}-fold evaluation)"
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
        return "### Check the blood pressure value (allowed: 50–300)."

    if not 50 <= numeric_inputs["Cholesterol"] <= 700:
        return "### Check the cholesterol value (allowed: 50–700)."

    if not 20 <= numeric_inputs["Heart Rate"] <= 300:
        return "### Check the heart rate value (allowed: 20–300)."

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
        predicted_class_index = list(model.classes_).index(prediction)
        model_score = probabilities[predicted_class_index] * 100
        score_line = (
            f"**Model score for this predicted class:** {model_score:.1f}%"
        )
    except Exception:
        score_line = "**Model score:** unavailable"

    if prediction.lower() == "disease":
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

**Important medical disclaimer:** This is an educational software
demonstration, not a medical diagnosis or a clinically validated risk
estimate. The model score is not a measure of medical certainty.
Please consult a qualified healthcare professional for health concerns.
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
# 6. PREMIUM GRADIO INTERFACE
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
> diagnostic tool. The quality of its predictions depends on the
> training data.
""")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Personal Information")

            age = gr.Number(
                label="Age (years)", value=45,
                minimum=1, maximum=120, precision=0
            )

            sex = gr.Dropdown(
                ["Male", "Female"],
                label="Sex", value="Male"
            )

            chest_pain = gr.Dropdown(
                ["None", "Mild", "Moderate", "Severe"],
                label="Chest Pain Category", value="Mild"
            )

        with gr.Column():
            gr.Markdown("### 🩺 Health Information")

            blood_pressure = gr.Number(
                label="Blood Pressure (dataset value)", value=130
            )

            cholesterol = gr.Number(
                label="Cholesterol (dataset value)", value=220
            )

            heart_rate = gr.Number(
                label="Heart Rate (dataset value)", value=150
            )

            ecg_results = gr.Dropdown(
                ["Normal", "Abnormal"],
                label="ECG Results", value="Normal"
            )

            blood_sugar = gr.Dropdown(
                ["Normal", "High"],
                label="Blood Sugar", value="Normal"
            )

    with gr.Row():
        predict_button = gr.Button(
            "🔍 Predict", variant="primary", elem_id="predict-button"
        )
        reset_button = gr.Button("↺ Reset Form")

    result = gr.Markdown(label="Prediction Result")

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
# 7. START SERVER / PUBLIC LINK
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("HEART DISEASE DETECTION - SEMINAR 4")
    print("=" * 55)
    print(f"Usable dataset records: {len(df)}")
    print(evaluation_text)

    # Set GRADIO_SHARE=true to create a temporary public link.
    # For permanent hosting, deploy on Hugging Face Spaces.
    public_share=True

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=public_share
    )