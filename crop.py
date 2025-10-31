import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------
# 🌱 App Title
# ---------------------------------------------
st.set_page_config(page_title="Crop Yield Prediction", layout="centered")
st.title("🌾 Crop Field Data Prediction using Linear Regression")

# ---------------------------------------------
# 📂 File Upload or Sample Data
# ---------------------------------------------
st.sidebar.header("Upload or Use Sample Data")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
else:
    st.info("Using sample dataset (Rainfall, Temperature, Soil pH, Nitrogen, Phosphorus, Potassium)")
    df = pd.DataFrame({
        'Rainfall': np.random.randint(100, 300, 50),
        'Temperature': np.random.randint(25, 35, 50),
        'Humidity': np.random.randint(60, 90, 50),
        'Soil_pH': np.round(np.random.uniform(5.5, 7.5, 50), 2),
        'Nitrogen': np.random.randint(30, 60, 50),
        'Phosphorus': np.random.randint(20, 40, 50),
        'Potassium': np.random.randint(15, 35, 50),
        'Yield': np.random.randint(2500, 5000, 50)
    })

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------------
# ⚙️ Feature Selection
# ---------------------------------------------
features = [col for col in df.columns if col != 'Yield']
target = 'Yield'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# ---------------------------------------------
# 📈 Model Performance
# ---------------------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📉 Model Evaluation")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# ---------------------------------------------
# 🌿 Suggest Crop Function (Multiple Suggestions)
# ---------------------------------------------
def suggest_crop(yield_kg):
    """Suggest up to 3 suitable crops based on predicted yield range."""
    crop_ranges = [
        (2500, "Millet 🌾"),
        (3000, "Maize 🌽"),
        (3500, "Wheat 🌿"),
        (4000, "Paddy 🌾"),
        (4500, "Sugarcane 🍬"),
        (5000, "Cotton 🧵"),
        (5500, "Banana 🍌")
    ]

    suggestions = []
    for limit, crop in crop_ranges:
        if yield_kg <= limit:
            idx = crop_ranges.index((limit, crop))
            # Suggest up to 3 crops starting from this range
            suggestions = [c[1] for c in crop_ranges[idx:idx+3]]
            break

    # If higher than all ranges, suggest top 3 high-yield crops
    if not suggestions:
        suggestions = [c[1] for c in crop_ranges[-3:]]

    return suggestions

# ---------------------------------------------
# 🌱 Prediction Form (Interactive)
# ---------------------------------------------
st.subheader("🌱 Predict Yield for New Crop Field")

col1, col2, col3 = st.columns(3)

with col1:
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, value=200.0)
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, value=28.0)

with col2:
    humidity = st.number_input("💨 Humidity (%)", min_value=0.0, value=75.0)
    soil_ph = st.number_input("🌍 Soil pH", min_value=0.0, max_value=14.0, value=6.5)

with col3:
    nitrogen = st.number_input("🧪 Nitrogen (kg/ha)", min_value=0.0, value=45.0)
    phosphorus = st.number_input("🧪 Phosphorus (kg/ha)", min_value=0.0, value=30.0)
    potassium = st.number_input("🧪 Potassium (kg/ha)", min_value=0.0, value=20.0)

if st.button("🔍 Predict Yield and Suggest Crops"):
    new_data = pd.DataFrame({
        'Rainfall': [rainfall],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Soil_pH': [soil_ph],
        'Nitrogen': [nitrogen],
        'Phosphorus': [phosphorus],
        'Potassium': [potassium]
    })

    # Predict yield using trained model
    prediction = model.predict(new_data)[0]

    # Suggest up to 3 crops
    crop_suggestions = suggest_crop(prediction)

    st.success(f"📈 **Predicted Yield:** {prediction:.2f} kg/ha")
    st.info(f"🌾 **Suggested Crops:**\n{',\n '.join(crop_suggestions)}")

# ---------------------------------------------
# 📉 Show actual vs predicted
# ---------------------------------------------
st.subheader("📊 Actual vs Predicted Yield")
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
st.dataframe(results.head(10))

# ---------------------------------------------
# 📈 Graphs: Actual vs Predicted
# ---------------------------------------------
st.subheader("📊 Actual vs Predicted Yield Graphs")

# Scatter Plot
fig1, ax1 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel("Actual Yield")
ax1.set_ylabel("Predicted Yield")
ax1.set_title("Actual vs Predicted Yield (Scatter Plot)")
st.pyplot(fig1)

# Line Plot for first 20 records
fig2, ax2 = plt.subplots()
ax2.plot(y_test.values[:20], label='Actual', marker='o')
ax2.plot(y_pred[:20], label='Predicted', marker='x')
ax2.set_xlabel("Actual Yield")
ax2.set_ylabel("Predicted Yield")
ax2.set_title("Actual vs Predicted Yield (Line Plot - First 20 Samples)")
ax2.legend()
st.pyplot(fig2)

st.subheader("📊 Yield Distribution Bar Chart")

# Define yield ranges
bins = [0, 3000, 3500, 4000, 4500, 5000, np.inf]
labels = ["<3000", "3000-3500", "3500-4000", "4000-4500", "4500-5000", ">5000"]

# Categorize yields
df['Yield_Range'] = pd.cut(df['Yield'], bins=bins, labels=labels, include_lowest=True)

# Count per category
yield_counts = df['Yield_Range'].value_counts().sort_index()

# Plot bar chart
fig, ax = plt.subplots()
sns.barplot(x=yield_counts.index, y=yield_counts.values, palette="pastel", ax=ax)
ax.set_xlabel("Yield Range (kg/ha)")
ax.set_ylabel("Number of Fields")
ax.set_title("Yield Distribution by Range")
st.pyplot(fig)

# st.subheader("📊 Yield Distribution Pie Chart")

# # Define yield ranges
# bins = [0, 3000, 3500, 4000, 4500, 5000, np.inf]
# labels = ["<3000", "3000-3500", "3500-4000", "4000-4500", "4500-5000", ">5000"]

# # Categorize yields
# df['Yield_Range'] = pd.cut(df['Yield'], bins=bins, labels=labels, include_lowest=True)

# # Count per category
# yield_counts = df['Yield_Range'].value_counts().sort_index()

# # Pie chart
# fig, ax = plt.subplots()
# ax.pie(yield_counts, labels=yield_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
# ax.set_title("Yield Distribution by Range")
# st.pyplot(fig)
