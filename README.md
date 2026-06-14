# 🍔 McDonald's Energy Calculator - Streamlit Web App

A machine learning-powered web application that **predicts calorie content (Energy kCal)** based on nutritional information for McDonald's India menu items.

## 📋 Overview

This Streamlit application uses a **Regression Model (XGBRegressor)** to predict the energy/calorie content of menu items based on their nutritional composition. Users can:
- Input nutritional values manually to predict calories
- Upload CSV files for batch calorie predictions
- Use quick preset examples for demonstration

## 🎯 Model Information

- **Model Type**: XGBRegressor (Regression)
- **Training Data**: McDonald's India Menu nutritional dataset
- **Target Variable**: Energy (kCal) - Calorie Content
- **Input Features**: 9 nutritional parameters
- **Model Performance**: R² Score = 0.9347 (93.47% accuracy)

### Input Features
1. Protein (g)
2. Total fat (g)
3. Saturated fat (g)
4. Trans fat (g)
5. Cholesterol (mg)
6. Total carbohydrate (g)
7. Total sugars (g)
8. Added sugars (g)
9. Sodium (mg)

### Output
- **Energy (kCal)**: Predicted calorie content per serving
- **Energy Category**: Low (<300 kCal), Medium (300-600 kCal), High (>600 kCal)

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── mcd_model.pkl            # Trained XGBRegressor model
├── train_model.py           # Script to train model from data
├── preprocessor.py          # Data preprocessing utilities
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .streamlit/
    └── config.toml          # Streamlit configuration
```

## 🚀 Installation & Setup

### 1. **Clone/Download the Project**
```bash
cd path/to/mcd-app
```

### 2. **Create Virtual Environment (Recommended)**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## 🎯 Running the Application

### Local Development
```bash
streamlit run app.py
```

The app will open at: `http://localhost:8501`

### Deployment (Cloud)

#### **Streamlit Cloud** (Free & Easy)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Select the file: `app.py`
5. Deploy!

#### **Heroku** (Alternative)
1. Create `Procfile`:
   ```
   web: streamlit run app.py --logger.level=error
   ```
2. Push to Heroku Git
3. Deploy!

#### **AWS / Google Cloud / Azure**
- Deploy Docker container or use their ML services
- Ensure `mcd_model.pkl` is included in deployment

## 📖 Usage Guide

### **Method 1: Manual Nutritional Input**
- Fill in nutritional values (protein, fat, carbs, etc.)
- Click "Predict Energy (kCal)"
- View predicted calorie content and energy category

**Example:**
- Input: Protein 20g, Fat 15g, Carbs 50g, etc.
- Output: 452 kCal 🟠 Medium Energy

### **Method 2: CSV Batch Upload**
- Prepare CSV with nutritional columns (see feature list above)
- Upload file → Get batch predictions
- View statistics (average, min, max energy)
- Download results as CSV

**CSV Format:**
```
Protein (g),Total fat (g),Sat Fat (g),...,Sodium (mg)
10,15,5,...,700
20,20,8,...,850
```

### **Method 3: Quick Preset Examples**
- Select preset menu items (Salad, Burger, Wrap, etc.)
- Get instant energy predictions
- Perfect for understanding model behavior

## ⚙️ Configuration

Edit `.streamlit/config.toml` to customize:
- Theme (light/dark)
- Page layout
- Sidebar state
- Font and colors

## 📊 Model Details

### Training Process
1. Load McDonald's India Menu dataset
2. Handle missing values (median imputation)
3. Apply log transformation to specific features
4. Detect and clip outliers (IQR method)
5. Train XGBRegressor with:
   - n_estimators=100
   - learning_rate=0.1
   - max_depth=3
   - random_state=42
6. Achieve R² = 0.9347 on test set

### Performance Metrics
- **R² Score**: 0.9347 (93.47% of variance explained)
- **Model**: XGBRegressor
- **Train/Test Split**: 80/20

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: mcd_model.pkl` | Ensure model file is in app directory. Run `python train_model.py` to generate it. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| App won't start | Check Python version (3.8+) and dependencies |
| Slow predictions | Model caching is enabled; first run may be slower |

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1 | Web framework |
| pandas | 2.1.3 | Data handling |
| numpy | 1.24.3 | Numerical operations |
| scikit-learn | 1.3.2 | ML utilities |
| xgboost | >=1.6 | Gradient boosting model |
| joblib | 1.3.2 | Model serialization |

## 🎨 Features

✅ Manual nutritional input with instant predictions  
✅ Batch CSV processing for multiple items  
✅ Quick preset examples for testing  
✅ Energy categorization (Low/Medium/High)  
✅ Prediction statistics (avg, min, max)  
✅ CSV export functionality  
✅ Responsive UI with custom styling  
✅ Error handling & validation  
✅ High-accuracy regression model (R² = 0.9347)

## 🔐 Security Notes

- Model file (`mcd_model.pkl`) contains no personal data
- Predictions computed locally; no external API calls
- File uploads are temporary and deleted after processing
- No data is stored on server

## 📝 Retraining the Model

To retrain the model with new data:

```bash
python train_model.py
```

This script:
1. Loads the McDonald's India Menu dataset
2. Applies preprocessing (outlier detection, log transformation)
3. Trains XGBRegressor on nutritional features
4. Saves model as `mcd_model.pkl`
5. Reports R² Score

## 📧 Support

For issues or questions, refer to:
- [Streamlit Documentation](https://docs.streamlit.io)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [Scikit-learn Documentation](https://scikit-learn.org)

---

**Last Updated**: 2026  
**Model Type**: Regression (Energy Prediction)  
**Status**: Production-Ready ✅

