import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Accident Data Analysis",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🚗 Road Accident Data Analysis")
st.write(
    "Exploratory Data Analysis of the Indian Road Accident Dataset"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("indian_roads_dataset.csv")


df = load_data()

# ============================================================
# DATA CLEANING
# ============================================================

# Missing values before cleaning
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

# Remove festival because 99.42% values are missing
if "festival" in df.columns:
    df = df.drop(columns=["festival"])

# Remove duplicate rows
df = df.drop_duplicates()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Dashboard")

page = st.sidebar.selectbox(
    "Select Analysis",
    [
        "Overview",
        "Data Cleaning",
        "Statistics",
        "Categorical Analysis",
        "Distribution Analysis",
        "Correlation Analysis"
    ]
)

# ============================================================
# OVERVIEW
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
        "Data Type": df.dtypes.astype(str),
        "Non-Null Values": df.notnull().sum().values
    })

    st.dataframe(
        info_data,
        width="stretch"
    )

# ============================================================
# DATA CLEANING
# ============================================================

elif page == "Data Cleaning":

    st.header("🧹 Data Cleaning")

    st.subheader("Missing Values Before Cleaning")

    if len(missing_values) > 0:

        missing_table = pd.DataFrame({
            "Missing Values": missing_values,
            "Percentage": (
                missing_values / 20000 * 100
            ).round(2)
        })

        st.dataframe(
            missing_table,
            width="stretch"
        )

    else:
        st.success("No missing values found.")

    st.subheader("Cleaning Operations")

    st.write("✔ Removed the `festival` column because most values were missing.")

    st.write("✔ Checked and removed duplicate rows.")

    col1, col2 = st.columns(2)

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

    st.subheader("Final Dataset")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

# ============================================================
# STATISTICS
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
        "Skewness helps identify whether a distribution is symmetric "
        "or shifted toward one side."
    )

    st.info(
        "Kurtosis describes the shape and tail behavior of a distribution."
    )

# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

elif page == "Categorical Analysis":

    st.header("📊 Categorical Data Analysis")

    column = st.selectbox(
        "Select a categorical variable",
        [
            "accident_severity",
            "day_of_week",
            "weather",
            "road_type",
            "traffic_density",
            "cause",
            "visibility",
            "city",
            "state"
        ]
    )

    counts = df[column].value_counts()

    st.subheader(
        f"Distribution of {column.replace('_', ' ').title()}"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    # Limit very large categories
    if len(counts) > 15:
        counts = counts.head(15)

    sns.barplot(
        x=counts.values,
        y=counts.index,
        ax=ax
    )

    ax.set_xlabel("Number of Accidents")
    ax.set_ylabel(
        column.replace("_", " ").title()
    )

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
        "Count": df[column].value_counts(),
        "Percentage": percentage
    })

    st.dataframe(
        table,
        width="stretch"
    )

# ============================================================
# DISTRIBUTION ANALYSIS
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
        df[selected_column],
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

    # Statistics for selected variable

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

    # Boxplot

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
# CORRELATION ANALYSIS
# ============================================================

elif page == "Correlation Analysis":

    st.header("🔗 Correlation Analysis")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    correlation = df[numeric_columns].corr()

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        ax=ax
    )

    ax.set_title(
        "Correlation between Numerical Variables"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Interpretation")

    st.write("""
    Correlation measures the relationship between two numerical variables.

    • A value close to +1 indicates a strong positive relationship.

    • A value close to -1 indicates a strong negative relationship.

    • A value close to 0 indicates a weak or no linear relationship.
    """)

# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.write("🚗 Road Accident Data Analysis")
st.sidebar.write("B.Tech CSE - AI")