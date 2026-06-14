import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="McDonald's Energy Calculator",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""<style>.main { padding: 2rem; }</style>""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("mcd_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file 'mcd_model.pkl' not found.")
        st.stop()

def get_energy_category(energy):
    """Categorize energy level"""
    if energy <= 100:
        return "Low Energy 💚", "#51cf66"
    elif energy <= 500:
        return "Moderate Energy 🟠", "#ffa500"
    else:
        return "High Energy 🔴", "#ff6b6b"

def preprocess_features(protein, total_fat, sat_fat, trans_fat, cholesterol, carbs, sugars, added_sugars, sodium):
    """Apply same transformations as training: log1p for protein and total_fat"""
    # Apply log transformation to Protein and Total fat (same as training)
    protein_log = np.log1p(protein)
    total_fat_log = np.log1p(total_fat)
    
    # Return in exact order: Protein, Total fat, Sat Fat, Trans fat, Cholesterols, Carbs, Sugars, Added Sugars, Sodium
    return np.array([[
        protein_log,      # Log transformed
        total_fat_log,    # Log transformed
        sat_fat,          # Raw
        trans_fat,        # Raw
        cholesterol,      # Raw
        carbs,            # Raw
        sugars,           # Raw
        added_sugars,     # Raw
        sodium            # Raw
    ]])

# Main app
st.title("🍔 McDonald's Energy Calculator")
st.markdown("**Predict Calorie Content (Energy kCal) Based on Nutritional Values**")
st.markdown("---")

model = load_model()

# Sidebar
st.sidebar.title("⚙️ Configuration")
input_method = st.sidebar.radio(
    "Select Input Method:",
    ["📋 Fill Nutritional Values", "📊 Upload CSV", "🎯 Quick Examples"]
)
st.sidebar.markdown("---")
st.sidebar.info("💡 Enter nutritional values to predict calorie content")

# ============ METHOD 1: Manual Input ============
if input_method == "📋 Fill Nutritional Values":
    st.subheader("📝 Enter Nutritional Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        protein = st.number_input("Protein (g)", min_value=0.0, step=0.5)
        total_fat = st.number_input("Total fat (g)", min_value=0.0, step=0.5)
        sat_fat = st.number_input("Sat Fat (g)", min_value=0.0, step=0.5)
    
    with col2:
        trans_fat = st.number_input("Trans fat (g)", min_value=0.0, step=0.1)
        cholesterol = st.number_input("Cholesterols (mg)", min_value=0.0, step=5.0)
        carbs = st.number_input("Total carbohydrate (g)", min_value=0.0, step=0.5)
    
    with col3:
        sugars = st.number_input("Total Sugars (g)", min_value=0.0, step=0.5)
        added_sugars = st.number_input("Added Sugars (g)", min_value=0.0, step=0.5)
        sodium = st.number_input("Sodium (mg)", min_value=0.0, step=10.0)
    
    if st.button("⚡ Predict Energy (kCal)", key="predict_btn"):
        try:
            # Apply preprocessing (log transformation to inputs only)
            input_data = preprocess_features(protein, total_fat, sat_fat, trans_fat, cholesterol, carbs, sugars, added_sugars, sodium)
            
            # Model predicts actual energy value (not log-transformed)
            predicted_energy = int(round(model.predict(input_data)[0]))
            category_text, color = get_energy_category(predicted_energy)
            
            st.markdown("---")
            st.success("✅ Prediction Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    f"<div style='background-color: {color}; padding: 2rem; border-radius: 1rem; text-align: center;'>"
                    f"<h2>⚡ Predicted Energy</h2>"
                    f"<h1>{predicted_energy} kCal</h1>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                st.metric("Energy Category", category_text)
                st.markdown("### 📋 Input Summary")
                st.write(f"• Protein: {protein}g")
                st.write(f"• Fat: {total_fat}g")
                st.write(f"• Carbs: {carbs}g")
                st.write(f"• Cholesterol: {cholesterol}mg")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============ METHOD 2: CSV Upload ============
elif input_method == "📊 Upload CSV":
    st.subheader("📁 Upload CSV File")
    st.write("CSV should have columns: Protein (g), Total fat (g), Sat Fat (g), Trans fat (g), Cholesterols (mg), Total carbohydrate (g), Total Sugars (g), Added Sugars (g), Sodium (mg)")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📋 Data Preview:")
            st.dataframe(df.head())
            
            if st.button("⚡ Predict Energy for All Rows", key="predict_csv_btn"):
                feature_cols = ['Protein (g)', 'Total fat (g)', 'Sat Fat (g)', 'Trans fat (g)', 
                               'Cholesterols (mg)', 'Total carbohydrate (g)', 'Total Sugars (g)', 
                               'Added Sugars (g)', 'Sodium (mg)']
                
                if all(col in df.columns for col in feature_cols):
                    # Apply log transformation to Protein and Total fat
                    df_processed = df.copy()
                    df_processed['Protein (g)'] = np.log1p(df_processed['Protein (g)'])
                    df_processed['Total fat (g)'] = np.log1p(df_processed['Total fat (g)'])
                    
                    X = df_processed[feature_cols].values
                    predictions = model.predict(X)
                    # No inverse transform needed - Energy is not log-transformed
                    predictions_int = [int(round(p)) for p in predictions]
                    
                    results_df = df.copy()
                    results_df['Predicted_Energy_kCal'] = predictions_int
                    results_df['Energy_Category'] = [
                        get_energy_category(int(round(p)))[0] for p in predictions
                    ]
                    
                    st.success("✅ Predictions Complete!")
                    st.dataframe(results_df)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Energy", f"{int(np.mean(predictions_int))} kCal")
                    with col2:
                        st.metric("Min Energy", f"{int(np.min(predictions_int))} kCal")
                    with col3:
                        st.metric("Max Energy", f"{int(np.max(predictions_int))} kCal")
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="energy_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("❌ CSV missing required columns")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============ METHOD 3: Quick Examples ============
elif input_method == "🎯 Quick Examples":
    st.subheader("🎯 Quick Preset Examples")
    
    # Sample data - raw values (will be log-transformed in preprocess)
    examples = {
        "Light Salad": [8, 5, 1, 0, 0, 25, 4, 0, 400],
        "Veggie Burger": [10, 14, 5, 0.2, 2.5, 57, 8, 4.5, 706],
        "McSpicy Paneer": [20, 39, 17, 0.2, 22, 52, 8, 5, 1075],
        "Chicken Wrap": [18, 22, 8, 0.3, 30, 48, 6, 2, 850],
        "High Protein": [35, 25, 10, 0.2, 50, 40, 3, 1, 750],
    }
    
    selected_example = st.selectbox("Select a menu item:", list(examples.keys()))
    
    if st.button("⚡ Predict Energy", key="predict_example_btn"):
        try:
            values = examples[selected_example]
            input_data = preprocess_features(*values)
            predicted_energy = int(round(model.predict(input_data)[0]))
            
            category_text, color = get_energy_category(predicted_energy)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    f"<div style='background-color: {color}; padding: 2rem; border-radius: 1rem; text-align: center;'>"
                    f"<h2>⚡ Predicted Energy</h2>"
                    f"<h1>{predicted_energy} kCal</h1>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                st.metric("Energy Category", category_text)
                st.markdown(f"**Example:** {selected_example}")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🍔 McDonald's Energy Calculator | XGBRegressor (R² = 0.94)"
    "</div>",
    unsafe_allow_html=True
)
