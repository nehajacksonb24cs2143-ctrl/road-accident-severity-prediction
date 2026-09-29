import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import normaltest, probplot


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Accident Statistical Analysis",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CUSTOM PAGE TITLE
# ============================================================

st.title("🚗 Road Accident Statistical Analysis Dashboard")

st.markdown(
    """
    ### Exploratory Data Analysis of Indian Road Accident Data

    This dashboard focuses on understanding the statistical behavior
    of road accident data through data cleaning, descriptive statistics,
    distribution analysis, normality analysis, categorical analysis,
    and correlation analysis.
    """
)

st.info(
    "Project Focus: Exploratory Data Analysis and Statistical "
    "Behavior of Indian Road Accident Data"
)


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

@st.cache_data
def load_original_data():

    data = pd.read_csv("indian_roads_dataset.csv")

    return data


original_df = load_original_data()


# ============================================================
# DATA CLEANING
# ============================================================

# Keep the original data for demonstrating missing values
df = original_df.copy()

# Remove festival because it contains approximately
# 99.42% missing values
if "festival" in df.columns:
    df = df.drop(columns=["festival"])


# ============================================================
# COLUMN INFORMATION
# ============================================================

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include="object"
).columns.tolist()


# ============================================================
# VARIABLES FOR DISTRIBUTION ANALYSIS
# ============================================================

analysis_columns = [
    "latitude",
    "longitude",
    "hour",
    "lanes",
    "temperature",
    "vehicles_involved",
    "casualties",
    "risk_score"
]

# Make sure all selected columns exist
analysis_columns = [
    column
    for column in analysis_columns
    if column in df.columns
]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Dashboard Navigation")

section = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "Data Cleaning",
        "Descriptive Statistics",
        "Distribution Analysis",
        "Normality Analysis",
        "Categorical Analysis",
        "Correlation Analysis"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if section == "Overview":

    st.header("📊 Road Accident Dataset Overview")

    st.markdown(
        """
        The dataset contains information about road accidents,
        including location, time, road conditions, weather,
        traffic conditions, casualties, vehicles involved,
        accident severity, and risk score.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Total Features",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            f"{df.isnull().sum().sum():,}"
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            f"{df.duplicated().sum():,}"
        )

    st.divider()

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader("🔍 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # NUMERICAL AND CATEGORICAL VARIABLES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🔢 Numerical Variables")

        numerical_summary = pd.DataFrame({
            "Variable": numerical_columns
        })

        st.dataframe(
            numerical_summary,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        st.subheader("🔤 Categorical Variables")

        categorical_summary = pd.DataFrame({
            "Variable": categorical_columns
        })

        st.dataframe(
            categorical_summary,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # ACCIDENT SEVERITY
    # --------------------------------------------------------

    st.subheader("🚦 Accident Severity Distribution")

    severity_counts = df[
        "accident_severity"
    ].value_counts()

    severity_table = pd.DataFrame({
        "Severity": severity_counts.index,
        "Count": severity_counts.values,
        "Percentage": (
            severity_counts.values /
            len(df) * 100
        ).round(2)
    })

    col1, col2 = st.columns(2)

    with col1:

        st.dataframe(
            severity_table,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        sns.countplot(
            data=df,
            x="accident_severity",
            order=severity_counts.index,
            ax=ax
        )

        ax.set_title(
            "Accident Severity Distribution"
        )

        ax.set_xlabel(
            "Accident Severity"
        )

        ax.set_ylabel(
            "Number of Accidents"
        )

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# DATA CLEANING
# ============================================================

elif section == "Data Cleaning":

    st.header("🧹 Data Cleaning")

    st.markdown(
        """
        This section shows the quality of the original dataset
        and the cleaning decisions applied before analysis.
        """
    )

    # --------------------------------------------------------
    # ORIGINAL DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("Original Dataset")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Original Rows",
            f"{len(original_df):,}"
        )

    with col2:

        st.metric(
            "Original Columns",
            original_df.shape[1]
        )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    st.subheader("Missing Value Analysis")

    missing_df = pd.DataFrame({
        "Column": original_df.columns,
        "Missing Values": original_df.isnull().sum().values,
        "Missing Percentage": (
            original_df.isnull().sum().values /
            len(original_df) * 100
        ).round(2)
    })

    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ]

    if len(missing_df) > 0:

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No missing values were found."
        )

    # --------------------------------------------------------
    # FESTIVAL COLUMN
    # --------------------------------------------------------

    if "festival" in original_df.columns:

        festival_missing = original_df[
            "festival"
        ].isnull().sum()

        festival_percentage = (
            festival_missing /
            len(original_df) * 100
        )

        st.warning(
            f"The 'festival' column contains "
            f"{festival_missing:,} missing values "
            f"({festival_percentage:.2f}%). "
            f"It was removed because the majority of "
            f"its values are missing."
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    st.subheader("Duplicate Records")

    duplicate_count = original_df.duplicated().sum()

    if duplicate_count == 0:

        st.success(
            "No duplicate records were found."
        )

    else:

        st.warning(
            f"{duplicate_count:,} duplicate records were found."
        )

    # --------------------------------------------------------
    # CLEANED DATASET
    # --------------------------------------------------------

    st.subheader("Cleaned Dataset")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Rows After Cleaning",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Columns After Cleaning",
            df.shape[1]
        )

    st.subheader("Cleaned Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

elif section == "Descriptive Statistics":

    st.header("📊 Descriptive Statistics")

    st.markdown(
        """
        Descriptive statistics summarize the central tendency,
        spread, and shape of numerical variables.
        """
    )

    # --------------------------------------------------------
    # VARIABLE SELECTION
    # --------------------------------------------------------

    selected_variable = st.selectbox(
        "Select a numerical variable",
        analysis_columns
    )

    data = df[
        selected_variable
    ].dropna()

    # --------------------------------------------------------
    # CALCULATE STATISTICS
    # --------------------------------------------------------

    mean_value = data.mean()
    median_value = data.median()
    std_value = data.std()
    variance_value = data.var()
    min_value = data.min()
    max_value = data.max()
    skewness_value = data.skew()
    kurtosis_value = data.kurt()

    # --------------------------------------------------------
    # STATISTICS CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Mean",
            f"{mean_value:.3f}"
        )

    with col2:

        st.metric(
            "Median",
            f"{median_value:.3f}"
        )

    with col3:

        st.metric(
            "Std Dev",
            f"{std_value:.3f}"
        )

    with col4:

        st.metric(
            "Variance",
            f"{variance_value:.3f}"
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:

        st.metric(
            "Minimum",
            f"{min_value:.3f}"
        )

    with col6:

        st.metric(
            "Maximum",
            f"{max_value:.3f}"
        )

    with col7:

        st.metric(
            "Skewness",
            f"{skewness_value:.3f}"
        )

    with col8:

        st.metric(
            "Kurtosis",
            f"{kurtosis_value:.3f}"
        )

    st.divider()

    # --------------------------------------------------------
    # COMPLETE STATISTICAL TABLE
    # --------------------------------------------------------

    st.subheader(
        "Complete Statistical Summary"
    )

    statistics = pd.DataFrame({
        "Mean": df[analysis_columns].mean(),
        "Median": df[analysis_columns].median(),
        "Std Dev": df[analysis_columns].std(),
        "Variance": df[analysis_columns].var(),
        "Minimum": df[analysis_columns].min(),
        "Maximum": df[analysis_columns].max(),
        "Skewness": df[analysis_columns].skew(),
        "Kurtosis": df[analysis_columns].kurt()
    })

    st.dataframe(
        statistics.round(3),
        use_container_width=True
    )


# ============================================================
# DISTRIBUTION ANALYSIS
# ============================================================

elif section == "Distribution Analysis":

    st.header("📈 Distribution Analysis")

    st.markdown(
        """
        Histograms show the frequency distribution of a numerical
        variable, while the KDE curve provides a smooth estimate
        of its distribution.
        """
    )

    # --------------------------------------------------------
    # VARIABLE SELECTION
    # --------------------------------------------------------

    selected_variable = st.selectbox(
        "Select a numerical variable",
        analysis_columns
    )

    data = df[
        selected_variable
    ].dropna()

    # --------------------------------------------------------
    # HISTOGRAM + KDE
    # --------------------------------------------------------

    st.subheader(
        f"Distribution of {selected_variable}"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.histplot(
        data=data,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_variable}"
    )

    ax.set_xlabel(
        selected_variable
    )

    ax.set_ylabel(
        "Frequency"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # BOX PLOT
    # --------------------------------------------------------

    st.subheader(
        f"Box Plot of {selected_variable}"
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 3)
    )

    sns.boxplot(
        x=data,
        ax=ax2
    )

    ax2.set_xlabel(
        selected_variable
    )

    st.pyplot(fig2)

    plt.close(fig2)

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    skew = data.skew()

    if abs(skew) < 0.5:

        interpretation = (
            "The variable is approximately symmetric "
            "based on its skewness."
        )

    elif skew >= 0.5:

        interpretation = (
            "The variable shows positive/right skewness."
        )

    else:

        interpretation = (
            "The variable shows negative/left skewness."
        )

    st.info(
        f"Skewness = {skew:.3f}. {interpretation}"
    )


# ============================================================
# NORMALITY ANALYSIS
# ============================================================

elif section == "Normality Analysis":

    st.header("📐 Normality Analysis")

    st.markdown(
        """
        Normality analysis checks whether a numerical variable
        approximately follows a normal distribution.
        """
    )

    # --------------------------------------------------------
    # VARIABLE SELECTION
    # --------------------------------------------------------

    selected_variable = st.selectbox(
        "Select a numerical variable",
        analysis_columns
    )

    data = df[
        selected_variable
    ].dropna()

    # --------------------------------------------------------
    # NORMALITY TEST
    # --------------------------------------------------------

    statistic, p_value = normaltest(data)

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Test Statistic",
            f"{statistic:.4f}"
        )

    with col2:

        st.metric(
            "P-value",
            f"{p_value:.6f}"
        )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    if p_value > 0.05:

        st.success(
            "The test does not provide sufficient evidence "
            "to reject the assumption of normality."
        )

    else:

        st.warning(
            "The test indicates a statistically significant "
            "deviation from a normal distribution."
        )

    st.info(
        "Because the dataset contains 20,000 observations, "
        "the normality test can detect even small deviations. "
        "Therefore, the Q-Q plot and distribution shape should "
        "also be considered."
    )

    # --------------------------------------------------------
    # Q-Q PLOT
    # --------------------------------------------------------

    st.subheader(
        f"Q-Q Plot of {selected_variable}"
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    probplot(
        data,
        dist="norm",
        plot=ax
    )

    ax.set_title(
        f"Q-Q Plot of {selected_variable}"
    )

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

elif section == "Categorical Analysis":

    st.header("📋 Categorical Analysis")

    st.markdown(
        """
        Categorical analysis examines the frequency and percentage
        distribution of non-numerical variables.
        """
    )

    # --------------------------------------------------------
    # AVAILABLE CATEGORICAL VARIABLES
    # --------------------------------------------------------

    usable_categorical = [
        column
        for column in categorical_columns
        if df[column].nunique() <= 50
    ]

    selected_variable = st.selectbox(
        "Select a categorical variable",
        usable_categorical
    )

    # --------------------------------------------------------
    # FREQUENCY
    # --------------------------------------------------------

    frequency = df[
        selected_variable
    ].value_counts()

    percentage = (
        df[selected_variable]
        .value_counts(normalize=True) * 100
    ).round(2)

    summary = pd.DataFrame({
        "Frequency": frequency,
        "Percentage": percentage
    })

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        f"Distribution of {selected_variable}"
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    # --------------------------------------------------------
    # BAR CHART
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.countplot(
        data=df,
        x=selected_variable,
        order=frequency.index,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_variable}"
    )

    ax.set_xlabel(
        selected_variable
    )

    ax.set_ylabel(
        "Frequency"
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

elif section == "Correlation Analysis":

    st.header("🔗 Correlation Analysis")

    st.markdown(
        """
        Correlation measures the strength and direction of
        linear relationships between numerical variables.
        """
    )

    # --------------------------------------------------------
    # CORRELATION MATRIX
    # --------------------------------------------------------

    correlation_matrix = df[
        analysis_columns
    ].corr()

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title(
        "Correlation Matrix"
    )

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # CORRELATION TABLE
    # --------------------------------------------------------

    st.subheader(
        "Correlation Values"
    )

    st.dataframe(
        correlation_matrix.round(3),
        use_container_width=True
    )

    st.info(
        """
        Interpretation:
        +1 indicates a strong positive linear relationship,
        0 indicates little or no linear relationship,
        and -1 indicates a strong negative linear relationship.

        Correlation describes association and does not imply causation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Road Accident Statistical Analysis"
)

st.sidebar.caption(
    "Data Science Project"
)