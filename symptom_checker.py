# Streamlit frontend for Personalized Healthcare Recommendation System
import pandas as pd
import streamlit as st
import joblib
import requests
import math
from streamlit_js_eval import get_geolocation
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# -------------------------------
# Function to fetch nearby hospitals using OSM
# -------------------------------
def get_nearby_medical_places(lat, lon):
    overpass_url = "https://overpass-api.de/api/interpreter"

    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:5000,{lat},{lon});
      node["amenity"="clinic"](around:5000,{lat},{lon});
      node["healthcare"="diagnostic"](around:5000,{lat},{lon});
    );
    out;
    """

    response = requests.get(overpass_url, params={'data': query})
    data = response.json()

    places = []

    for element in data['elements']:
        name = element['tags'].get('name', 'Unnamed')
        place_lat = element['lat']
        place_lon = element['lon']

        distance = math.sqrt((lat - place_lat)**2 + (lon - place_lon)**2)

        places.append((name, distance))

    places.sort(key=lambda x: x[1])

    return places[:5]


# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load('random_forest_model.pkl')

# -------------------------------
# Load dataset to get symptom columns
# -------------------------------
df_symptoms = pd.read_excel('health_symptoms_dataset.xlsx')
symptom_columns = df_symptoms.drop('Disease', axis=1).columns.tolist()

# -------------------------------
# Recommendations dictionary
# -------------------------------
recommendations = {
    'VIRAL FEVER': {
        'Doctor': 'General Physician',
        'Tests': 'CBC (Complete Blood Count), Temperature check',
        'Recommendation': 'If fever persists for more than 3 days or chills/sweating are severe, see doctor immediately'
    },
    'DENGUE': {
        'Doctor': 'General Physician / Infectious Disease Specialist',
        'Tests': 'CBC (platelet count), Dengue NS1 antigen test',
        'Recommendation': 'High urgency! See doctor immediately if you have high fever, severe headache, joint pain, or rash'
    },
    'GASTRITIS': {
        'Doctor': 'Gastroenterologist',
        'Tests': 'Endoscopy (if chronic), Blood test for H. pylori',
        'Recommendation': 'See doctor if abdominal pain, acidity, nausea, or vomiting persist'
    },
    'DIABETES': {
        'Doctor': 'Endocrinologist',
        'Tests': 'Fasting Blood Sugar, HbA1c, Urine test',
        'Recommendation': 'Schedule a check-up soon if you have excessive thirst or frequent urination'
    },
    'MALARIA': {
        'Doctor': 'General Physician / Infectious Disease Specialist',
        'Tests': 'Peripheral blood smear, Rapid diagnostic test (RDT)',
        'Recommendation': 'High urgency! Seek medical care immediately if fever with chills and sweating occurs'
    }
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Personalized Healthcare Recommendation System")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #d7fcfc;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}

header {
    visibility: hidden;
}

.stApp {
    background-color: #d7fcfc;
}
</style>
""", unsafe_allow_html=True)


st.markdown(
   "<h1 style='text-align:center; margin-top:-30px;'>🩺 Personalized Healthcare Recommendation System</h1>",
    unsafe_allow_html=True
)

st.write("Select your symptoms below and click **Predict Disease**")

st.markdown("""
    <style>
    .disease-card {
        background-color: #eef8ff;   /* very light blue */
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #d6eaff;
        margin-bottom: 20px;
    }

    .disease-name {
        font-size: 22px;
        font-weight: 600;
        color: #1f4e79;
    }

    .disease-prob {
        font-size: 18px;
        color: #2c6fa3;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


# -------------------------------
# User symptom input (Responsive Grid)
# -------------------------------
user_input = {}

cols = st.columns(3)   

for i, symptom in enumerate(symptom_columns):
    with cols[i % 3]:
        user_input[symptom] = st.checkbox(symptom.replace("_", " "))


# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict Disease"):
    st.session_state.prediction_done = True
if st.session_state.prediction_done:

    # Check if at least one symptom is selected
    if sum(user_input.values()) == 0:
        st.warning("⚠️ Please select at least one symptom to get a prediction.")
        st.stop()

    input_df = pd.DataFrame([[1 if user_input[s] else 0 for s in symptom_columns]],
                            columns=symptom_columns)

    # Get probabilities
    probabilities = model.predict_proba(input_df)[0]
    diseases = model.classes_

    # Top 3 predictions
    top3 = sorted(
        zip(diseases, probabilities),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    top_disease, top_prob = top3[0]

# Confidence / uncertainty handling
    if top_prob < 0.5:
        st.warning(
            "⚠️ Symptoms overlap across multiple conditions. "
            "The prediction is uncertain. Please consult a doctor for proper diagnosis."
        )
    else:
        st.success(
            f"Most likely condition: {top_disease} ({top_prob*100:.2f}%)"
        )

    st.subheader("🧠 Possible Conditions Based on Symptoms")
    cols = st.columns(len(top3))

    for i, (disease, prob) in enumerate(top3):
        with cols[i]:

            st.markdown(f"""
            <div class="disease-card">
                <div class="disease-name">{disease.capitalize()}</div>
                <div class="disease-prob">{prob*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            rec = recommendations.get(disease.upper(), {})

            st.write(f"**Doctor:** {rec.get('Doctor', 'N/A')}")
            st.write(f"**Tests:** {rec.get('Tests', 'N/A')}")
            st.write(f"**Recommendation:** {rec.get('Recommendation', 'N/A')}")
            st.write("---")
    # -------------------------------
    # Auto Detect Location & Show Nearby Hospitals
    # -------------------------------
    st.subheader("📍 Detecting Your Location")

    location = get_geolocation()

    if location:
        lat = location["coords"]["latitude"]
        lon = location["coords"]["longitude"]

        st.success("Location detected successfully!")

        with st.spinner("Fetching nearby hospitals..."):
            nearby_places = get_nearby_medical_places(lat, lon)

        st.subheader("🏥 Nearby Hospitals & Diagnostic Centers")

        if nearby_places:
            for name, dist in nearby_places:
                st.markdown(f"""
                <div class="disease-card">
                    {name}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No nearby medical centers found.")
    else:
        st.info("Please allow location access to see nearby hospitals.")

