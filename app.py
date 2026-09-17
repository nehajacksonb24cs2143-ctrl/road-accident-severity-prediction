import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Accident Analysis",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚗 Road Accident Analysis & Severity Prediction")

st.markdown(
    """
    **Interactive dashboard for analysing road accidents and
    predicting accident severity using Machine Learning.**
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("indian_roads_dataset.csv")

    return df


try:

    df = load_data()

except Exception:

    st.error(
        "❌ Could not load Road.csv. "
        "Make sure Road.csv is in the same folder as app.py."
    )

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ============================================================
# BASIC CLEANING
# ============================================================

df = df.drop_duplicates()


for column in df.columns:

    if df[column].isna().any():

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):

            df[column] = df[column].fillna(
                df[column].median()
            )

        # Categorical columns
        else:

            mode_value = df[column].mode()

            if not mode_value.empty:

                df[column] = df[column].fillna(
                    mode_value.iloc[0]
                )

            else:

                df[column] = df[column].fillna(
                    "Unknown"
                )


# ============================================================
# CHECK TARGET
# ============================================================

if "accident_severity" not in df.columns:

    st.error(
        "❌ The column 'accident_severity' was not found "
        "in Road.csv."
    )

    st.write("Available columns:")

    st.write(df.columns.tolist())

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 Dashboard Filters")

st.sidebar.write(
    "Use the filters below to explore the accident data."
)


# ============================================================
# SEVERITY FILTER
# ============================================================

severity_options = sorted(
    df["accident_severity"]
    .dropna()
    .astype(str)
    .unique()
)


selected_severity = st.sidebar.multiselect(
    "Accident Severity",
    severity_options,
    default=severity_options
)


# ============================================================
# WEATHER FILTER
# ============================================================

if "weather" in df.columns:

    weather_options = sorted(
        df["weather"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_weather = st.sidebar.multiselect(
        "Weather",
        weather_options,
        default=weather_options
    )

else:

    selected_weather = []


# ============================================================
# TRAFFIC DENSITY FILTER
# ============================================================

if "traffic_density" in df.columns:

    traffic_options = sorted(
        df["traffic_density"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_traffic = st.sidebar.multiselect(
        "Traffic Density",
        traffic_options,
        default=traffic_options
    )

else:

    selected_traffic = []


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if selected_severity:

    filtered_df = filtered_df[
        filtered_df["accident_severity"]
        .astype(str)
        .isin(selected_severity)
    ]


if "weather" in df.columns and selected_weather:

    filtered_df = filtered_df[
        filtered_df["weather"]
        .astype(str)
        .isin(selected_weather)
    ]


if "traffic_density" in df.columns and selected_traffic:

    filtered_df = filtered_df[
        filtered_df["traffic_density"]
        .astype(str)
        .isin(selected_traffic)
    ]


# ============================================================
# DASHBOARD KPIs
# ============================================================

st.header("📊 Accident Overview")


col1, col2, col3, col4 = st.columns(4)


# Total accidents
with col1:

    st.metric(
        "Total Accidents",
        f"{len(filtered_df):,}"
    )


# Fatal accidents
with col2:

    fatal_count = (
        filtered_df["accident_severity"]
        .astype(str)
        .str.lower()
        .eq("fatal")
        .sum()
    )

    st.metric(
        "Fatal Accidents",
        f"{fatal_count:,}"
    )


# Casualties
with col3:

    if "casualties" in filtered_df.columns:

        total_casualties = pd.to_numeric(
            filtered_df["casualties"],
            errors="coerce"
        ).sum()

    else:

        total_casualties = 0

    st.metric(
        "Total Casualties",
        f"{total_casualties:,.0f}"
    )


# Vehicles
with col4:

    if "vehicles_involved" in filtered_df.columns:

        total_vehicles = pd.to_numeric(
            filtered_df["vehicles_involved"],
            errors="coerce"
        ).sum()

    else:

        total_vehicles = 0

    st.metric(
        "Vehicles Involved",
        f"{total_vehicles:,.0f}"
    )


st.divider()


# ============================================================
# SEVERITY DISTRIBUTION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("Accident Severity Distribution")

    severity_count = (
        filtered_df["accident_severity"]
        .value_counts()
        .reset_index()
    )

    severity_count.columns = [
        "Severity",
        "Count"
    ]

    fig = px.pie(
        severity_count,
        names="Severity",
        values="Count",
        hole=0.45,
        title="Severity Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TRAFFIC DENSITY
# ============================================================

with col2:

    if "traffic_density" in filtered_df.columns:

        st.subheader("Traffic Density")

        traffic_count = (
            filtered_df["traffic_density"]
            .value_counts()
            .reset_index()
        )

        traffic_count.columns = [
            "Traffic Density",
            "Accidents"
        ]

        fig = px.bar(
            traffic_count,
            x="Traffic Density",
            y="Accidents",
            text="Accidents",
            title="Accidents by Traffic Density"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# ACCIDENTS BY HOUR
# ============================================================

if "hour" in filtered_df.columns:

    st.subheader("⏰ Accidents by Hour")

    hour_data = pd.to_numeric(
        filtered_df["hour"],
        errors="coerce"
    )

    hourly = (
        hour_data
        .value_counts()
        .sort_index()
        .reset_index()
    )

    hourly.columns = [
        "Hour",
        "Accidents"
    ]

    fig = px.line(
        hourly,
        x="Hour",
        y="Accidents",
        markers=True,
        title="Accidents by Hour of Day"
    )

    fig.update_xaxes(
        dtick=1
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TEMPERATURE
# ============================================================

if "temperature" in filtered_df.columns:

    st.subheader("🌡️ Temperature Distribution")

    temperature_data = pd.to_numeric(
        filtered_df["temperature"],
        errors="coerce"
    ).dropna()

    fig = px.histogram(
        temperature_data,
        x=temperature_data,
        nbins=30,
        title="Accidents by Temperature"
    )

    fig.update_layout(
        xaxis_title="Temperature",
        yaxis_title="Number of Accidents"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# VEHICLES INVOLVED
# ============================================================

if "vehicles_involved" in filtered_df.columns:

    st.subheader("🚘 Vehicles Involved")

    vehicle_data = pd.to_numeric(
        filtered_df["vehicles_involved"],
        errors="coerce"
    )

    vehicle_count = (
        vehicle_data
        .value_counts()
        .sort_index()
        .reset_index()
    )

    vehicle_count.columns = [
        "Vehicles Involved",
        "Accidents"
    ]

    fig = px.bar(
        vehicle_count,
        x="Vehicles Involved",
        y="Accidents",
        text="Accidents",
        title="Accidents by Number of Vehicles"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TOP ACCIDENT CAUSES
# ============================================================

if "cause" in filtered_df.columns:

    st.subheader("⚠️ Major Causes of Accidents")

    cause_count = (
        filtered_df["cause"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    cause_count.columns = [
        "Cause",
        "Accidents"
    ]

    fig = px.bar(
        cause_count.sort_values("Accidents"),
        x="Accidents",
        y="Cause",
        orientation="h",
        title="Top 10 Accident Causes"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# LOCATION MAP
# ============================================================

if (
    "latitude" in filtered_df.columns
    and "longitude" in filtered_df.columns
):

    st.subheader("🗺️ Accident Locations")

    map_df = filtered_df[
        [
            "latitude",
            "longitude"
        ]
    ].copy()

    map_df["latitude"] = pd.to_numeric(
        map_df["latitude"],
        errors="coerce"
    )

    map_df["longitude"] = pd.to_numeric(
        map_df["longitude"],
        errors="coerce"
    )

    map_df = map_df.dropna()

    if not map_df.empty:

        fig = px.scatter_map(
            map_df,
            lat="latitude",
            lon="longitude",
            zoom=5,
            height=500,
            title="Geographical Distribution of Accidents"
        )

        fig.update_layout(
            map_style="open-street-map"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MACHINE LEARNING
# ============================================================

st.divider()

st.header("🤖 Machine Learning — Decision Tree")


# ============================================================
# PREPARE DATA
# ============================================================

ml_df = df.copy()


# Target
y = ml_df["accident_severity"].copy()


# Features
X_raw = ml_df.drop(
    columns=["accident_severity"]
)


# Remove ID-like columns
columns_to_drop = [
    "id",
    "accident_id"
]


X_raw = X_raw.drop(
    columns=[
        col for col in columns_to_drop
        if col in X_raw.columns
    ],
    errors="ignore"
)


# ============================================================
# HANDLE DATETIME COLUMNS
# ============================================================

for column in X_raw.columns:

    if (
        "date" in column
        or column == "time"
    ):

        try:

            converted = pd.to_datetime(
                X_raw[column],
                errors="coerce"
            )

            X_raw[column + "_year"] = (
                converted.dt.year
            )

            X_raw[column + "_month"] = (
                converted.dt.month
            )

            X_raw[column + "_day"] = (
                converted.dt.day
            )

            X_raw = X_raw.drop(
                columns=[column]
            )

        except Exception:

            pass


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

X = pd.get_dummies(
    X_raw,
    drop_first=True
)


# Convert everything to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# Fill any remaining missing values
X = X.fillna(0)


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y.astype(str)
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


# ============================================================
# DECISION TREE MODEL
# ============================================================

model = DecisionTreeClassifier(

    random_state=42,

    class_weight="balanced"
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Model",
        "Decision Tree"
    )


with col2:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col3:

    st.metric(
        "Weighted F1 Score",
        f"{f1 * 100:.2f}%"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("📌 Top 10 Important Features")


feature_importance = pd.Series(

    model.feature_importances_,

    index=X.columns

).sort_values(
    ascending=False
).head(10)


importance_df = (
    feature_importance
    .sort_values()
    .reset_index()
)


importance_df.columns = [
    "Feature",
    "Importance"
]


importance_df["Importance"] *= 100


fig = px.bar(

    importance_df,

    x="Importance",

    y="Feature",

    orientation="h",

    text="Importance",

    title="Top 10 Feature Importance"
)


fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)


fig.update_layout(
    xaxis_title="Importance (%)",
    yaxis_title="Feature"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# PREDICTION SECTION
# ============================================================

st.divider()

st.header("🔮 Predict Accident Severity")


st.write(
    "Enter the accident conditions below and click "
    "**Predict Accident Severity**."
)


col1, col2, col3 = st.columns(3)


# ============================================================
# INPUT 1
# ============================================================

with col1:

    risk_score = st.number_input(
        "Risk Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )


    casualties = st.number_input(
        "Casualties",
        min_value=0,
        max_value=100,
        value=2,
        step=1
    )


    vehicles_involved = st.number_input(
        "Vehicles Involved",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )


# ============================================================
# INPUT 2
# ============================================================

with col2:

    latitude = st.number_input(
        "Latitude",
        value=8.5241,
        format="%.4f"
    )


    longitude = st.number_input(
        "Longitude",
        value=76.9366,
        format="%.4f"
    )


    temperature = st.number_input(
        "Temperature",
        value=25.0,
        step=0.5
    )


# ============================================================
# INPUT 3
# ============================================================

with col3:

    hour = st.slider(
        "Hour of Accident",
        min_value=0,
        max_value=23,
        value=12
    )


    lanes = st.number_input(
        "Number of Lanes",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🚨 Predict Accident Severity",
    use_container_width=True
):

    # Create an empty row with the same
    # features used during model training

    input_data = pd.DataFrame(

        np.zeros(
            (1, len(X.columns))
        ),

        columns=X.columns
    )


    # Values entered by user

    feature_values = {

        "risk_score":
            risk_score,

        "casualties":
            casualties,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "temperature":
            temperature,

        "hour":
            hour,

        "vehicles_involved":
            vehicles_involved,

        "lanes":
            lanes
    }


    # Insert values where matching
    # model features exist

    for feature, value in feature_values.items():

        if feature in input_data.columns:

            input_data.loc[
                0,
                feature
            ] = value


    # Make prediction

    prediction = model.predict(
        input_data
    )


    predicted_label = (
        label_encoder
        .inverse_transform(prediction)[0]
    )


    # Display result

    st.success(
        f"### Predicted Accident Severity: "
        f"{predicted_label.upper()}"
    )


    # ========================================================
    # PREDICTION PROBABILITY
    # ========================================================

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            input_data
        )[0]


        probability_df = pd.DataFrame({

            "Severity":
                label_encoder.classes_,

            "Probability":
                probabilities * 100

        })


        fig = px.bar(

            probability_df,

            x="Severity",

            y="Probability",

            text="Probability",

            title="Prediction Probability"

        )


        fig.update_traces(

            texttemplate="%{text:.2f}%",

            textposition="outside"

        )


        fig.update_layout(

            yaxis_title="Probability (%)",

            xaxis_title="Severity",

            yaxis_range=[
                0,
                100
            ]

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


# ============================================================
# DATA PREVIEW
# ============================================================

st.divider()

with st.expander(
    "📋 View Filtered Accident Data"
):

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Road Accident Analysis & Severity Prediction | "
    "B.Tech CSE (AI) Microproject"
)