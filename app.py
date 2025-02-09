import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Title of the app
st.title("Excel Data Preparation App")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Load the Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)

    # Display the raw data
    st.subheader("Raw Data")
    st.write(df)

    # Data Cleaning and Preprocessing
    st.subheader("Data Cleaning and Preprocessing")

    # Handle missing values
    st.write("### Handle Missing Values")
    if df.isnull().sum().sum() > 0:
        st.write("Missing values detected:")
        st.write(df.isnull().sum())

        # Option to drop or fill missing values
        missing_value_option = st.radio(
            "Choose how to handle missing values:",
            ("Drop rows with missing values", "Fill missing values with mean/median/mode")
        )

        if missing_value_option == "Drop rows with missing values":
            df = df.dropna()
            st.write("Rows with missing values dropped.")
        else:
            fill_method = st.selectbox(
                "Choose fill method:",
                ("Mean", "Median", "Mode")
            )
            if fill_method == "Mean":
                df = df.fillna(df.mean())
            elif fill_method == "Median":
                df = df.fillna(df.median())
            elif fill_method == "Mode":
                df = df.fillna(df.mode().iloc[0])
            st.write(f"Missing values filled with {fill_method}.")

        st.write("Data after handling missing values:")
        st.write(df)
    else:
        st.write("No missing values found.")

    # Encode categorical variables
    st.write("### Encode Categorical Variables")
    categorical_columns = df.select_dtypes(include=["object"]).columns
    if len(categorical_columns) > 0:
        st.write("Categorical columns detected:")
        st.write(categorical_columns)

        # Option to encode categorical variables
        if st.checkbox("Encode categorical variables using Label Encoding"):
            label_encoder = LabelEncoder()
            for col in categorical_columns:
                df[col] = label_encoder.fit_transform(df[col])
            st.write("Categorical variables encoded.")
            st.write(df)
    else:
        st.write("No categorical columns found.")

    # Display final cleaned data
    st.subheader("Final Cleaned Data")
    st.write(df)

    # Option to download the cleaned data
    st.write("### Download Cleaned Data")
    cleaned_file = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Cleaned Data as CSV",
        data=cleaned_file,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )
else:
    st.write("Please upload an Excel file to get started.")