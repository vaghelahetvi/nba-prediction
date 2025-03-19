# Import required libraries
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model and encoders
print("🔍 Loading trained model and encoders...")
model = joblib.load("nba_player_predictor.pkl")
label_encoder_players = joblib.load("label_encoder_players.pkl")
label_encoder_teams = joblib.load("label_encoder_teams.pkl")

# Load test dataset
print("📂 Loading test dataset...")
test_data_path = "NBA_test.csv"
test_labels_path = "NBA_test_labels.csv"

test_data = pd.read_csv(test_data_path)
test_labels = pd.read_csv(test_labels_path)

# Display column names for debugging
print(f"🔍 Test Data Columns: {test_data.columns.tolist()}")
print(f"🔍 Test Labels Columns: {test_labels.columns.tolist()}")

# Rename the removed player column if necessary
if "removed_value" not in test_labels.columns:
    print("⚠️ Missing 'removed_value' column. Please check the labels file.")
    exit(1)

# Drop unnecessary columns early to reduce memory usage
columns_to_keep = ['season', 'home_team', 'away_team', 'starting_min',
                   'home_0', 'home_1', 'home_2', 'home_3',
                   'away_0', 'away_1', 'away_2', 'away_3', 'away_4']

test_data = test_data[columns_to_keep]

# Convert categorical features to reduce memory usage
for col in ['home_team', 'away_team', 'home_0', 'home_1', 'home_2', 'home_3',
            'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
    test_data[col] = test_data[col].astype("category")

# Encode categorical variables safely
def safe_encode_column(column, encoder):
    """Safely encode categorical data, replacing unknown values with -1."""
    return column.apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

# Encode each column individually to avoid mismatched lengths
for col in ['home_0', 'home_1', 'home_2', 'home_3',
            'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
    try:
        test_data[col] = safe_encode_column(test_data[col], label_encoder_players)
    except Exception as e:
        print(f"⚠️ Encoding error in {col}: {e}")

test_data['home_team'] = safe_encode_column(test_data['home_team'], label_encoder_teams)
test_data['away_team'] = safe_encode_column(test_data['away_team'], label_encoder_teams)

# Verify column consistency
print(f"✅ Test data shape after encoding: {test_data.shape}")

# Predict missing players
print(f"🔄 Running predictions on {len(test_data)} test cases...")
predicted_players = model.predict(test_data)

# Convert back to original player names
predicted_players = label_encoder_players.inverse_transform(predicted_players)

# Save predictions
predictions_df = pd.DataFrame({
    "season": test_data["season"],
    "Match #": range(1, len(predicted_players) + 1),
    "Predicted Player": predicted_players,
    "Actual Player": test_labels["removed_value"]
})

# Check accuracy
predictions_df["Match Status"] = predictions_df.apply(lambda row: "✅ Match" if row["Predicted Player"] == row["Actual Player"] else "❌ Mismatch", axis=1)

# Calculate yearly accuracy
yearly_stats = predictions_df.groupby("season")["Match Status"].value_counts().unstack().fillna(0)
yearly_stats["Total Matches"] = yearly_stats["✅ Match"] + yearly_stats["❌ Mismatch"]
yearly_stats["Accuracy (%)"] = (yearly_stats["✅ Match"] / yearly_stats["Total Matches"]) * 100

# Save results
predictions_df.to_csv("Results/NBA_test_predictions.csv", index=False)
yearly_stats.to_csv("Results/NBA_test_yearly_accuracy.csv")

# Print Summary
matches_correct = predictions_df["Match Status"].value_counts().get("✅ Match", 0)
accuracy = (matches_correct / len(test_data)) * 100
print("\n📁 Predictions saved to 'NBA_test_predictions.csv'.")
print("📊 Yearly accuracy saved to 'NBA_test_yearly_accuracy.csv'.\n")
print(f"📊 **Final Accuracy on Test Data:** {accuracy:.2f}%")
print("\n📅 **Yearly Breakdown:**\n")
print(yearly_stats)

print("🏁 Process Completed Successfully!")
