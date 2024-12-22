import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Configuring Streamlit
st.set_page_config(page_title="Used Car Dataset Analysis", layout="wide")

# Title and Description
st.title("Used Car Dataset Analysis")
st.write("""
This app performs data preprocessing and visualization on the **Used Car Dataset**.
Upload the dataset file `used_car_dataset.csv` to begin.
""")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Data Wrangling
    st.subheader("Data Wrangling")

    # Drop unnecessary columns
    st.write("Removing unnecessary columns: `Age`, `PostedDate`, `AdditionInfo`")
    df.drop(['Age', 'PostedDate', 'AdditionInfo'], axis=1, inplace=True)

    # Handle Missing Values
    st.write("Filling missing values in `kmDriven` with the mean.")
    df['kmDriven'] = df['kmDriven'].astype(str).str.replace('km', '').str.replace(',', '').astype(float)
    df['AskPrice'] = df['AskPrice'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['kmDriven'] = df['kmDriven'].fillna(df['kmDriven'].mean())

    # Remove duplicates
    st.write("Removing duplicate rows.")
    initial_shape = df.shape
    df = df.drop_duplicates()
    st.write(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

    # Encoding categorical columns
    st.write("Encoding categorical columns: `Transmission`, `FuelType`, `Owner`.")
    le = LabelEncoder()
    df['transmission_encoded'] = le.fit_transform(df['Transmission'])
    df['fueltype_encoded'] = le.fit_transform(df['FuelType'])
    df['owner_encoded'] = le.fit_transform(df['Owner'])

    # Normalize numeric columns
    st.write("Normalizing numeric columns: `kmDriven` and `AskPrice`.")
    scaler = MinMaxScaler()
    df[['kmDriven', 'AskPrice']] = scaler.fit_transform(df[['kmDriven', 'AskPrice']])

    st.subheader("Processed Dataset")
    st.write(df.head())

    # Data Availability Checking and Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    # Data Visualization
    st.subheader("Data Visualizations")

    # Correlation Matrix
    st.write("### Correlation Matrix")
    correlation = df[['kmDriven', 'AskPrice', 'transmission_encoded', 'owner_encoded', 'fueltype_encoded']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Distribution of `kmDriven`
    st.write("### Distribution of `kmDriven`")
    fig, ax = plt.subplots()
    sns.histplot(df['kmDriven'], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

    # Distribution of `AskPrice`
    st.write("### Distribution of `AskPrice`")
    fig, ax = plt.subplots()
    sns.histplot(df['AskPrice'], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot by `FuelType`
    st.write("### Boxplot of `AskPrice` by `FuelType`")
    fig, ax = plt.subplots()
    sns.boxplot(x='FuelType', y='AskPrice', data=df, ax=ax)
    st.pyplot(fig)

    # Average `AskPrice` by Transmission
    st.write("### Average `AskPrice` by Transmission")
    fig, ax = plt.subplots()
    sns.barplot(x='Transmission', y='AskPrice', data=df, ax=ax)
    st.pyplot(fig)

    # Outlier Detection
    st.write("### Outlier Detection")
    st.write("Boxplot of `kmDriven` and `AskPrice` to detect outliers.")
    fig, ax = plt.subplots()
    sns.boxplot(df['kmDriven'], ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(df['AskPrice'], ax=ax)
    st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to proceed.")
