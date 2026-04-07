# Budget Shock Simulator

The Budget Shock Simulator is an end to end machine learning project that predicts monthly financial stress using a student spending dataset with 1,000 records. The project began with data cleaning, feature engineering, and target creation, followed by training and evaluating a logistic regression model to estimate financial stress risk. It was then turned into an interactive Streamlit app where users can enter their own budget inputs, view predicted stress probability, risk level, and budget breakdown, and test custom what if scenarios. The final result is a deployed, shareable tool that combines machine learning, financial analysis, and interactive decision support in one practical application.

## Features
- Predicts monthly financial stress probability
- Classifies risk level based on user inputs
- Shows a clear budget breakdown
- Supports custom what if shock scenarios
- Includes combined shock testing for multiple budget changes
- Presents results with cleaner labels, tables, charts, and key insights

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## Files
- `app.py` — Streamlit application
- `model_training.ipynb` — model development, cleaning, feature engineering, and evaluation
- `budget_stress_model.pkl` — trained logistic regression model
- `budget_stress_scaler.pkl` — saved scaler for preprocessing
- `requirements.txt` — project dependencies
