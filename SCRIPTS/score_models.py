"""
===============================================================================
File: score_models.py
Description:
    This script loads trained Isolation Forest models and a test dataset of
    parking tickets (year 2024). It scales the test features, predicts anomalies,
    and evaluates model performance. For this dataset, all test instances are
    positive tickets. Predictions map anomaly (-1) to "no ticket" and normal (1)
    to "ticket". Accuracy is calculated for each model, stored in a csv called model_performance.csv,
    with a corresponding bar graph stored in model_performance.png.

Usage:
    python score_isolation_forest.py

Inputs:
    - ../DATA/Final/encoded_parking_tickets.csv : encoded numeric dataset
    - ../OUTPUT/Final/models/ : directory containing trained models and scalers

Outputs:
    - Printed DataFrame showing accuracy of each model on 2024 test set
    - A bar graph of model accuracies saved as ../OUTPUT/Final/model_performance.png
===============================================================================
"""

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load test data
# -------------------------------
test_file = "./DATA/Final/encoded_parking_tickets.csv"
df_test = pd.read_csv(test_file)

# Only use 2024 for testing
df_test = df_test[df_test['Year'] == 2024]

if df_test.empty:
    raise ValueError("No 2024 data found for testing!")

# -------------------------------
# 2. Define features for model
# -------------------------------
# Use all numeric columns except 'Year'
features = df_test.columns.tolist()
features.remove('Year')

X_test = df_test[features]

# -------------------------------
# 3. Load trained models
# -------------------------------
model_dir = "./OUTPUT/Final/models"

# Identify all Isolation Forest model files
model_files = [
    f for f in os.listdir(model_dir)
    if f.startswith("isolation_forest_") and f.endswith(".joblib")
]

# -------------------------------
# 4. Score each model
# -------------------------------
results = []

for model_file in model_files:
    # Extract label from filename (e.g., "2015-2019")
    label = model_file.replace("isolation_forest_", "").replace(".joblib", "")

    # Load the trained model and its scaler
    model = joblib.load(os.path.join(model_dir, model_file))
    scaler = joblib.load(os.path.join(model_dir, f"scaler_{label}.joblib"))

    # Scale test features using the trained scaler
    X_scaled = scaler.transform(X_test)

    # Predict anomalies
    # Isolation Forest returns:
    #   1  -> normal (inlier)
    #  -1  -> anomaly (outlier)
    preds = model.predict(X_scaled)

    # Map to ticket prediction:
    #   -1 (anomaly) -> 0 (no ticket)
    #    1 (normal)  -> 1 (ticket)
    y_pred = (preds == 1).astype(int)

    # True labels for test set: all 1 (ticket occurred)
    y_true = pd.Series([1] * len(df_test))

    # -------------------------------
    # 4a. Compute performance metrics
    # -------------------------------
    acc = accuracy_score(y_true, y_pred)

    # Store results, only need accuracy because the others are irrelevant when our dataset only contains positives
    results.append({
        'Model': label,
        'Accuracy': acc,
    })

# -------------------------------
# 5. Display results
# -------------------------------
results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

print("\n--- Model performance on 2024 test set ---")
print(results_df)

# save the csv to final output folder
results_df.to_csv('./OUTPUT/Final/model_performance.csv', index=False)

# create and save the bar graph to final output folder
graph = results_df.plot.bar(
    x='Model',
    y='Accuracy',
    legend=False,
    ylim=(0.75, 1),
    title='Isolation Forest Model Performance on 2024 Test Set'
)

# Label the y-axis
plt.ylabel('Accuracy (%)')

# Convert tick labels to percentages
plt.gca().set_yticklabels([f'{y * 100:.0f}%' for y in plt.gca().get_yticks()])

# Add a horizontal goal line at 80%
plt.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Goal: 80%')

plt.legend()
plt.tight_layout()
plt.savefig('./OUTPUT/Final/model_performance.png', dpi=300, bbox_inches='tight')
plt.close()