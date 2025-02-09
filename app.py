import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Title of the app
st.title("Excel Data Preparation and Model Training App")

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

    # Model Training Section
    st.subheader("Model Training")

    # Check if the data has a target column
    target_column = st.selectbox("Select the target column for model training:", df.columns)
    if target_column:
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data into training and testing sets
        test_size = st.slider("Select test set size (percentage):", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Choose a model
        model_option = st.selectbox("Choose a model:", ["Random Forest Classifier"])

        if model_option == "Random Forest Classifier":
            model = RandomForestClassifier()

        # Train the model
        if st.button("Train Model"):
            st.write("Training the model...")
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Display model performance
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")

            # Option to save the trained model
            if st.button("Save Model"):
                model_filename = "trained_model.pkl"
                joblib.dump(model, model_filename)
                st.write(f"Model saved as {model_filename}")

else:
    st.write("Please upload an Excel file to get started.")