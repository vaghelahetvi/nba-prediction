# Import required libraries
import os
import pandas as pd
import numpy as np
import psutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import shap
import glob


# Function to print memory usage
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"🖥️ Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")


# Step 1: Load and Combine Data
def load_data(csv_files):
    """Load all CSV files into a single DataFrame."""
    dataframes = [pd.read_csv(file) for file in csv_files]
    data = pd.concat(dataframes, ignore_index=True)
    return data


# Step 2: Preprocess Data
def preprocess_data(data):
    """Preprocess the dataset: Filter features, handle missing values, encode categorical variables."""

    allowed_features = [
        'game', 'season', 'home_team', 'away_team', 'starting_min',
        'home_0', 'home_1', 'home_2', 'home_3', 'home_4',
        'away_0', 'away_1', 'away_2', 'away_3', 'away_4'
    ]

    data = data[allowed_features].copy()
    data.dropna(subset=['home_4'], inplace=True)  # Remove rows with missing target values

    # Encode categorical variables (players and teams)
    players = pd.concat([data[col] for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                                               'away_0', 'away_1', 'away_2', 'away_3', 'away_4']])
    le = LabelEncoder()
    le.fit(players)

    for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
        data[col] = le.transform(data[col])

    teams = pd.concat([data['home_team'], data['away_team']])
    le_teams = LabelEncoder()
    le_teams.fit(teams)

    data['home_team'] = le_teams.transform(data['home_team'])
    data['away_team'] = le_teams.transform(data['away_team'])

    # Optimize data types
    for col in data.columns:
        if data[col].dtype == 'int64':
            data[col] = data[col].astype('int32')
        elif data[col].dtype == 'float64':
            data[col] = data[col].astype('float32')

    return data, le, le_teams


# Step 3: Train the Model
def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    X_train = X_train.drop(columns=['game'])
    model = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt', random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# Step 4: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model using accuracy and classification report."""
    X_test = X_test.drop(columns=['game'])
    y_pred = model.predict(X_test)

    print(f"\n✅ **Model Accuracy:** {accuracy_score(y_test, y_pred) * 100:.2f}%")
    #print("\n📊 **Classification Report:**\n")
    #print(classification_report(y_test, y_pred, zero_division=0))


# Step 5: Make Predictions
def predict_player(model, le, le_teams, test_data):
    """Predict the optimal fifth player for the home team."""
    test_data = test_data.drop(columns=['game'])
    predicted_player = model.predict(test_data)
    return le.inverse_transform(predicted_player)


# Main Function
def main():
    print("🔍 Loading Data...")

    csv_files = glob.glob('data/*.csv')
    if not csv_files:
        raise FileNotFoundError("❌ No CSV files found in the 'data/' folder. Please check the path.")

    data = load_data(csv_files)
    data, le, le_teams = preprocess_data(data)

    print("📊 Data preprocessing completed.")

    # Split into training and testing sets
    X = data.drop(columns=['home_4'])
    y = data['home_4']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🎯 Training the Model...")
    model = train_model(X_train, y_train)

    print("🛠️ Evaluating Model Performance...")
    evaluate_model(model, X_test, y_test)

    # Save the Model
    joblib.dump(model, 'nba_player_predictor.pkl')
    joblib.dump(le, 'label_encoder_players.pkl')
    joblib.dump(le_teams, 'label_encoder_teams.pkl')

    print("\n💾 Model and Encoders saved successfully.")

    # Make Predictions
    test_data = pd.DataFrame({
        'game': [1],
        'season': [2008],
        'home_team': [le_teams.transform(['LAL'])[0]],
        'away_team': [le_teams.transform(['GSW'])[0]],
        'starting_min': [0],
        'home_0': [le.transform(['Derek Fisher'])[0]],
        'home_1': [le.transform(['Kobe Bryant'])[0]],
        'home_2': [le.transform(['Lamar Odom'])[0]],
        'home_3': [le.transform(['Luke Walton'])[0]],
        'away_0': [le.transform(['Al Harrington'])[0]],
        'away_1': [le.transform(['Baron Davis'])[0]],
        'away_2': [le.transform(['Kelenna Azubuike'])[0]],
        'away_3': [le.transform(['Matt Barnes'])[0]],
        'away_4': [le.transform(['Monta Ellis'])[0]]
    })

    predicted_player = predict_player(model, le, le_teams, test_data)
    print(f"\n🏀 **Predicted Fifth Player:** {predicted_player[0]}")


# Run the main function
if __name__ == "__main__":
    main()
