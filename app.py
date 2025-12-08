# import streamlit as st
# import pandas as pd
# import joblib

# # Load saved model, scaler, and expected columns
# model = joblib.load("knn_heart_model.pkl")
# scaler = joblib.load("heart_scaler.pkl")
# expected_columns = joblib.load("heart_columns.pkl")

# st.title("Heart Stroke Prediction by akarsh")
# st.markdown("Provide the following details to check your heart stroke risk:")

# # Collect user input
# age = st.slider("Age", 18, 100, 40)
# sex = st.selectbox("Sex", ["M", "F"])
# chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
# resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
# cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
# fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
# resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
# max_hr = st.slider("Max Heart Rate", 60, 220, 150)
# exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
# oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
# st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# # When Predict is clicked
# if st.button("Predict"):

#     # Create a raw input dictionary
#     raw_input = {
#         'Age': age,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_bs,
#         'MaxHR': max_hr,
#         'Oldpeak': oldpeak,
#         'Sex_' + sex: 1,
#         'ChestPainType_' + chest_pain: 1,
#         'RestingECG_' + resting_ecg: 1,
#         'ExerciseAngina_' + exercise_angina: 1,
#         'ST_Slope_' + st_slope: 1
#     }

#     # Create input dataframe
#     input_df = pd.DataFrame([raw_input])

#     # Fill in missing columns with 0s
#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # Reorder columns
#     input_df = input_df[expected_columns]

#     # Scale the input
#     scaled_input = scaler.transform(input_df)

#     # Make prediction
#     prediction = model.predict(scaled_input)[0]

#     # Show result
#     if prediction == 1:
#         st.error("⚠️ High Risk of Heart Disease")
#     else:
#         st.success("✅ Low Risk of Heart Disease")











import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Model, Scaler, Columns
# ----------------------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ----------------------------
# App Title
# ----------------------------
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This interactive tool helps you **assess your heart disease risk**  
based on important health indicators.
""")

st.info("⚠️ This app is for educational use only. Not medical advice.")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("📝 Enter Your Health Details")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["Atypical Angina", "Non-Anginal Pain", "Typical Angina", "Asymptomatic"])
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 60, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 80, 600, 200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.sidebar.selectbox("Resting ECG Result", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("ST Slope Type", ["Up", "Flat", "Down"])

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Predict Risk")

# ----------------------------
# Prediction Logic
# ----------------------------
if predict_btn:

    # Base continuous features
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
    }

    # One-hot encoding categorical values
    raw_input[f"Sex_{sex}"] = 1
    raw_input[f"ChestPainType_{chest_pain}"] = 1
    raw_input[f"RestingECG_{resting_ecg}"] = 1
    raw_input[f"ExerciseAngina_{exercise_angina}"] = 1
    raw_input[f"ST_Slope_{st_slope}"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add all missing model columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict class
    prediction = model.predict(scaled_input)[0]

    # Predict probability (if available)
    try:
        probability = model.predict_proba(scaled_input)[0][1] * 100
    except:
        probability = None

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    # ----------------------------
    # Show Results
    # ----------------------------
    if prediction == 1:
        if probability is not None:
            st.error(f"""🚨 **High Risk Detected!**
Your estimated risk probability is **{probability:.2f}%**.
Please consult a healthcare professional for a detailed evaluation.""")
        else:
            st.error("""🚨 **High Risk Detected!**
Please consult a healthcare professional for proper assessment.""")
    else:
        if probability is not None:
            st.success(f"""💚 **Low Risk Detected!**
Your estimated risk probability is **{probability:.2f}%**.
Maintain a healthy lifestyle!""")
        else:
            st.success("""💚 **Low Risk Detected!**
Keep maintaining a healthy lifestyle!""")

    st.markdown("---")
    st.caption("⚠️ Model predictions may not be accurate for all individuals. Always consult a doctor.")
