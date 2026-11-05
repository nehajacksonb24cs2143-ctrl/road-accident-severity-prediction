
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Accident Analysis and Prediction",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# 2. TITLE
# ============================================================

st.title("🚗 Road Accident Data Analysis & Severity Prediction")

st.write(
    "An interactive data science application for analyzing "
    "Indian road accident data and classifying accident severity."
)

st.divider()

# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    data = pd.read_csv("indian_roads_dataset.csv")
    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found! Please place "
        "'indian_roads_dataset.csv' in the project folder."
    )
    st.stop()

# ============================================================
# 4. DATA CLEANING
# ============================================================

# Record missing values before cleaning
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

# Remove festival column because most values are missing
if "festival" in df.columns:
    df = df.drop(columns=["festival"])

# Remove duplicate rows
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()

# Check required target column
if "accident_severity" not in df.columns:
    st.error("The dataset must contain accident_severity column.")
    st.stop()

# ============================================================
# 5. TRAIN MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    # Target variable
    y = data["accident_severity"]

    # Input features
    X = data.drop(
        columns=[
            "accident_severity",
            "accident_id",
            "date",
            "time",
            "festival",
            "risk_score"
        ],
        errors="ignore"
    )

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_transformer = SimpleImputer(
        strategy="median"
    )

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_transformer,
                numerical_columns
            ),
            (
                "cat",
                categorical_transformer,
                categorical_columns
            )
        ]
    )

    # Random Forest classifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    # Complete ML pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Train the model
    model.fit(X_train, y_train)

    return (
        model,
        X_test,
        y_test,
        X.columns.tolist(),
        numerical_columns,
        categorical_columns
    )


with st.spinner("Training Random Forest model..."):

    (
        model,
        X_test,
        y_test,
        feature_columns,
        numerical_columns,
        categorical_columns
    ) = train_model(df)

# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Dashboard")

page = st.sidebar.selectbox(
    "Select a Page",
    [
        "Overview",
        "Data Cleaning",
        "Statistics",
        "Categorical Analysis",
        "Distribution Analysis",
        "Correlation Analysis",
        "Prediction",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("🚗 Road Accident Project")
st.sidebar.write("B.Tech CSE - AI")

# ============================================================
# 7. OVERVIEW
# ============================================================

if page == "Overview":

    st.header("📋 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Total Features",
            len(df.columns)
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.subheader("Dataset Information")

    info_data = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Values": df.notnull().sum().values
    })

    st.dataframe(
        info_data,
        width="stretch"
    )

    st.subheader("Accident Severity Distribution")

    severity_counts = df["accident_severity"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x=severity_counts.index,
        y=severity_counts.values,
        ax=ax
    )

    ax.set_xlabel("Accident Severity")
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Distribution of Accident Severity")

    st.pyplot(fig)

# ============================================================
# 8. DATA CLEANING
# ============================================================

elif page == "Data Cleaning":

    st.header("🧹 Data Cleaning")

    st.subheader("Missing Values Before Cleaning")

    if len(missing_values) > 0:

        missing_table = pd.DataFrame({
            "Missing Values": missing_values,
            "Percentage": (
                missing_values / len(pd.read_csv(
                    "indian_roads_dataset.csv"
                )) * 100
            ).round(2)
        })

        st.dataframe(
            missing_table,
            width="stretch"
        )

    else:
        st.success("No missing values found.")

    st.subheader("Cleaning Operations")

    st.write(
        "✔ Removed the festival column because most values were missing."
    )

    st.write(
        "✔ Checked and removed duplicate records."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Records After Cleaning",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Features After Cleaning",
            len(df.columns)
        )

    with col3:
        st.metric(
            "Duplicate Rows Removed",
            duplicate_count
        )

    st.subheader("Final Dataset")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

# ============================================================
# 9. STATISTICS
# ============================================================

elif page == "Statistics":

    st.header("📈 Statistical Analysis")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    statistics = pd.DataFrame({
        "Mean": df[numeric_columns].mean(),
        "Median": df[numeric_columns].median(),
        "Std Dev": df[numeric_columns].std(),
        "Variance": df[numeric_columns].var(),
        "Minimum": df[numeric_columns].min(),
        "Maximum": df[numeric_columns].max(),
        "Skewness": df[numeric_columns].skew(),
        "Kurtosis": df[numeric_columns].kurt()
    })

    st.dataframe(
        statistics.round(3),
        width="stretch"
    )

    st.info(
        "Skewness measures the asymmetry of a distribution."
    )

    st.info(
        "Kurtosis describes the tail behavior of a distribution."
    )

# ============================================================
# 10. CATEGORICAL ANALYSIS
# ============================================================

elif page == "Categorical Analysis":

    st.header("📊 Categorical Data Analysis")

    available_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    column = st.selectbox(
        "Select a categorical variable",
        available_columns
    )

    counts = df[column].value_counts()

    display_counts = counts.head(15)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=display_counts.values,
        y=display_counts.index,
        ax=ax
    )

    ax.set_xlabel("Number of Accidents")
    ax.set_ylabel(column.replace("_", " ").title())

    ax.set_title(
        f"Accidents by {column.replace('_', ' ').title()}"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Frequency Table")

    percentage = (
        df[column]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    table = pd.DataFrame({
        "Count": counts,
        "Percentage": percentage
    })

    st.dataframe(
        table,
        width="stretch"
    )

# ============================================================
# 11. DISTRIBUTION ANALYSIS
# ============================================================

elif page == "Distribution Analysis":

    st.header("📉 Numerical Distribution Analysis")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    selected_column = st.selectbox(
        "Select a numerical variable",
        numeric_columns
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        df[selected_column].dropna(),
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_column}"
    )

    ax.set_xlabel(
        selected_column.replace("_", " ").title()
    )

    ax.set_ylabel("Frequency")

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Distribution Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean",
            round(df[selected_column].mean(), 3)
        )

    with col2:
        st.metric(
            "Median",
            round(df[selected_column].median(), 3)
        )

    with col3:
        st.metric(
            "Skewness",
            round(df[selected_column].skew(), 3)
        )

    with col4:
        st.metric(
            "Std Dev",
            round(df[selected_column].std(), 3)
        )

    st.subheader("Boxplot")

    fig, ax = plt.subplots(figsize=(9, 3))

    sns.boxplot(
        x=df[selected_column],
        ax=ax
    )

    ax.set_title(
        f"Boxplot of {selected_column}"
    )

    plt.tight_layout()

    st.pyplot(fig)

# ============================================================
# 12. CORRELATION ANALYSIS
# ============================================================

elif page == "Correlation Analysis":

    st.header("🔗 Correlation Analysis")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    correlation = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation between Numerical Variables"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Interpretation")

    st.write("""
    Correlation measures the strength and direction of a
    linear relationship between two numerical variables.

    • A value close to +1 indicates a positive relationship.

    • A value close to -1 indicates a negative relationship.

    • A value close to 0 indicates a weak linear relationship.

    Correlation does not necessarily mean causation.
    """)

# ============================================================
# 13. ACCIDENT SEVERITY PREDICTION
# ============================================================

elif page == "Prediction":

    st.header("🚗 Accident Severity Prediction")

    st.write(
        "Enter the accident details to classify the "
        "recorded accident severity."
    )

    st.info(
        "The prediction is generated using a trained "
        "Random Forest classification model."
    )

    user_input = {}

    st.subheader("Enter Accident Details")

    # Numerical inputs
    if numerical_columns:

        st.markdown("### Numerical Features")

        for column in numerical_columns:

            column_values = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna()

            if column_values.empty:
                continue

            minimum = float(column_values.min())
            maximum = float(column_values.max())
            default = float(column_values.median())

            if minimum == maximum:
                user_input[column] = minimum

            elif pd.api.types.is_integer_dtype(df[column]):

                user_input[column] = st.number_input(
                    column.replace("_", " ").title(),
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(default),
                    step=1
                )

            else:

                user_input[column] = st.number_input(
                    column.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=default
                )

    # Categorical inputs
    if categorical_columns:

        st.markdown("### Categorical Features")

        for column in categorical_columns:

            choices = sorted(
                df[column].dropna().unique().tolist(),
                key=str
            )

            if choices:

                user_input[column] = st.selectbox(
                    column.replace("_", " ").title(),
                    choices
                )

    st.divider()

    if st.button(
        "🔍 Predict Accident Severity",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [user_input],
            columns=feature_columns
        )

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        classes = model.named_steps[
            "classifier"
        ].classes_

        st.subheader("Prediction Result")

        severity = str(prediction).strip().lower()

        if severity == "fatal":

            st.error(
                f"🚨 Predicted Severity: {prediction}"
            )

        elif severity == "major":

            st.warning(
                f"⚠️ Predicted Severity: {prediction}"
            )

        else:

            st.success(
                f"✅ Predicted Severity: {prediction}"
            )

        st.subheader("Class Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Severity": classes,
            "Probability (%)": (
                probabilities * 100
            ).round(2)
        })

        st.dataframe(
            probability_df,
            hide_index=True,
            width="stretch"
        )

        st.bar_chart(
            probability_df.set_index("Severity")
        )

        st.caption(
            "Probabilities are model estimates, not guarantees "
            "or calibrated real-world accident probabilities."
        )

# ============================================================
# 14. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("📊 Random Forest Model Performance")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        width="stretch"
    )

    st.subheader("Confusion Matrix")

    classes = model.named_steps[
        "classifier"
    ].classes_

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )

    ax.set_xlabel("Predicted Severity")
    ax.set_ylabel("Actual Severity")
    ax.set_title("Random Forest Confusion Matrix")

    st.pyplot(fig)

    st.subheader("Model Information")

    st.write("Algorithm: Random Forest Classifier")
    st.write("Number of trees: 100")
    st.write("Training and testing split: 80:20")
    st.write("Class weighting: Balanced")

    st.write(
        "The model is evaluated using accuracy, precision, "
        "recall, F1-score and the confusion matrix."
    )

# ============================================================
# 15. FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Road Accident Data Analysis & Severity Prediction"
)

st.sidebar.caption(
    "B.Tech CSE - Artificial Intelligence"
)

st.markdown("---")

st.caption(
    "Academic project | Road Accident Data Analysis "
    "and Severity Classification"
)
