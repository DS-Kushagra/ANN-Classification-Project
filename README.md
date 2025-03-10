# Customer Churn Prediction

## Overview
Customer churn prediction is a machine learning-based solution that predicts the likelihood of a customer leaving a business. This project implements a deep learning model using **TensorFlow** to analyze customer data and determine their churn probability.

## Features
- Accepts user inputs such as **Geography, Gender, Age, Balance, Credit Score, Estimated Salary, Tenure, Number of Products, Credit Card Ownership, and Activity Status**.
- One-hot encodes categorical data for better machine learning model performance.
- Standardizes numerical features using a **pre-trained scaler**.
- Uses a **deep learning model (TensorFlow)** to predict customer churn probability.
- Provides an interactive **Streamlit-based UI** for easy user input and real-time predictions.

 **You can check out the app [here](https://customer-churn-model.streamlit.app/)**

## Tech Stack
- **Python**
- **Streamlit** (for web UI)
- **TensorFlow** (for deep learning model)
- **Scikit-learn** (for data preprocessing: Label Encoding, One-Hot Encoding, and Scaling)
- **Pandas & NumPy** (for data handling)
- **Pickle** (for saving and loading trained models and encoders)

## Installation & Setup

### Prerequisites
Ensure you have Python installed (recommended: Python 3.8+). You also need to install the following dependencies:

```bash
pip install streamlit tensorflow scikit-learn pandas numpy pickle-mixin
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### Run the Streamlit App
```bash
streamlit run app.py
```

## How It Works
1. The **trained deep learning model** (`model.h5`) is loaded.
2. The **label encoders and scaler** are loaded from pickle files.
3. The user inputs the required details through the Streamlit UI.
4. The categorical features (Geography, Gender) are encoded using **LabelEncoder** and **OneHotEncoder**.
5. The numerical features are scaled using the pre-trained **StandardScaler**.
6. The processed data is fed into the **deep learning model**, which predicts the probability of churn.
7. The output is displayed, indicating whether the customer is likely to churn or not.

## File Structure
```
📂 customer-churn-prediction
├── 📜 app.py                  # Streamlit web application
├── 📜 model.h5                # Pre-trained deep learning model
├── 📜 scaler.pkl              # StandardScaler for numerical feature transformation
├── 📜 encoder_geo.pkl         # OneHotEncoder for geography
├── 📜 LabelEncoder_Gender.pkl # LabelEncoder for gender
├── 📜 README.md               # Project documentation
```

## Future Improvements
- Enhancing the model with **hyperparameter tuning**.
- Incorporating **additional features** to improve prediction accuracy.
- Deploying the application on **Cloud (AWS/GCP/Heroku)** for broader access.
- Implementing a **database backend** to store user inputs and analyze trends over time.

## Author
**[Kushagra]** - *Machine Learning & Data Science Enthusiast*  
[LinkedIn](https://www.linkedin.com/in/kushagra--agrawal/) 

---

Feel free to contribute and improve this project!

