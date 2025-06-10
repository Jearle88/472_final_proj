import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# === Configuration ===
def load_team_ids_from_csv(file_path, column_name):
    """
    Load all unique team IDs from a given column in a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        column_name (str): Name of the column containing team IDs.

    Returns:
        list: A list of unique team IDs as integers.
    """
    df = pd.read_csv(file_path)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    team_ids = df[column_name].dropna().astype(int).unique().tolist()
    return team_ids
csv_path = "C:/Users/Johnt/OneDrive/Documents/GitHub/472_final_proj/all_teams_consolidated_data.csv"
TEAM_IDs = load_team_ids_from_csv("C:/Users/Johnt/OneDrive/Documents/GitHub/472_final_proj/archive/base_data/teams.csv", "teamId")
#TEAM_IDs.sort()  # Optional: ensures consistency in split

split_index = int(len(TEAM_IDs) * 0.8)

train_team_ids = TEAM_IDs[:split_index]   # Bottom 80%
test_team_ids = TEAM_IDs[split_index:]    # Top 20%
# === Load Data ===
df = pd.read_csv(csv_path)
print("=== Data Analysis ===")
print(f"Dataset shape: {df.shape}")
print(f"All columns: {list(df.columns)}")

# === SIMPLE: Just specify which columns are structural ===
# You need to manually set these to match your CSV
DATE_COLUMN = 'date'
HOME_TEAM_COLUMN = 'homeTeamId'  # Corrected from 'home_team_id'
AWAY_TEAM_COLUMN = 'awayTeamId'  # Corrected from 'away_team_id'

# Optional - if you have these columns
HOME_SCORE_COLUMN = 'homeTeamScore'  # Corrected from 'home_score'
AWAY_SCORE_COLUMN = 'awayTeamScore'  # Corrected from 'away_score'

print(f"\nUsing structural columns:")
print(f"Date: {DATE_COLUMN}")
print(f"Home Team: {HOME_TEAM_COLUMN}")
print(f"Away Team: {AWAY_TEAM_COLUMN}")
print(f"Home Score: {HOME_SCORE_COLUMN}")
print(f"Away Score: {AWAY_SCORE_COLUMN}")

# === Get ALL other columns as features ===
structural_columns = [DATE_COLUMN, HOME_TEAM_COLUMN, AWAY_TEAM_COLUMN]
if HOME_SCORE_COLUMN:
    structural_columns.append(HOME_SCORE_COLUMN)
if AWAY_SCORE_COLUMN:
    structural_columns.append(AWAY_SCORE_COLUMN)

# FEATURE COLUMNS = ALL columns except structural ones
#Feature_columns = [col for col in df.columns if col not in structural_columns]
# other test for fetarue cols
feature_columns=["possessionPct", "passPct", "totalShots", "shotsOnTarget","wonCorners","totalPasses"]
print(f"\nFEATURE COLUMNS (from CSV headings): {feature_columns}")
print(f"Number of feature columns: {len(feature_columns)}")

# Convert date and sort
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
df = df.sort_values(DATE_COLUMN)


# === Create Team Records ===
def create_team_records(df, team_ids):
    """Create records for teams using CSV column headings directly"""
    all_records = []

    for team_id in team_ids:
        print(f"\nProcessing Team {team_id}...")

        # Get all matches for this team
        team_matches = df[(df[HOME_TEAM_COLUMN] == team_id) | (df[AWAY_TEAM_COLUMN] == team_id)].copy()
        team_matches = team_matches.sort_values(DATE_COLUMN)

        print(f"Found {len(team_matches)} matches for team {team_id}")

        team_records = []

        for _, row in team_matches.iterrows():
            # Basic record info
            record = {
                'team_id': team_id,
                'date': row[DATE_COLUMN],
            }

            # Determine if home or away, and if won
            if row[HOME_TEAM_COLUMN] == team_id:
                # Team is home
                record['is_home'] = 1
                record['opponent_id'] = row[AWAY_TEAM_COLUMN]

                # Determine winner
                if HOME_SCORE_COLUMN and AWAY_SCORE_COLUMN:
                    home_score = row[HOME_SCORE_COLUMN] if pd.notna(row[HOME_SCORE_COLUMN]) else 0
                    away_score = row[AWAY_SCORE_COLUMN] if pd.notna(row[AWAY_SCORE_COLUMN]) else 0
                    record['won'] = 1 if home_score > away_score else 0
                else:
                    record['won'] = 0  # Default if no scores
            else:
                # Team is away
                record['is_home'] = 0
                record['opponent_id'] = row[HOME_TEAM_COLUMN]

                # Determine winner
                if HOME_SCORE_COLUMN and AWAY_SCORE_COLUMN:
                    home_score = row[HOME_SCORE_COLUMN] if pd.notna(row[HOME_SCORE_COLUMN]) else 0
                    away_score = row[AWAY_SCORE_COLUMN] if pd.notna(row[AWAY_SCORE_COLUMN]) else 0
                    record['won'] = 1 if away_score > home_score else 0
                else:
                    record['won'] = 0  # Default if no scores

            # ADD ALL FEATURE COLUMNS DIRECTLY FROM CSV
            for col in feature_columns:
                if col in row.index:
                    value = row[col]
                    # Convert to float if possible, otherwise 0
                    try:
                        record[col] = float(value) if pd.notna(value) else 0.0
                    except:
                        record[col] = 0.0
                else:
                    record[col] = 0.0

            team_records.append(record)

        if team_records:
            all_records.extend(team_records)

    return pd.DataFrame(all_records)


# === Create Training and Prediction Data ===
print("\n=== Creating Training Data ===")
training_data = create_team_records(df, train_team_ids)
print(f"Training data shape: {training_data.shape}")

print("\n=== Creating Prediction Data ===")
prediction_data = create_team_records(df, test_team_ids)
print(f"Prediction data shape: {prediction_data.shape}")

if len(training_data) == 0 or len(prediction_data) == 0:
    print("ERROR: No data found! Check your team IDs and column names.")
    exit()

# === Prepare Features ===
# Use: is_home + ALL your CSV feature columns
model_features = ['is_home'] + feature_columns
# #odel_features =  feature_columns

print(f"\nFeatures going into the model:")
for i, feature in enumerate(model_features, 1):
    print(f"{i:2d}. {feature}")


# Clean data - remove rows with missing target
training_clean = training_data.dropna(subset=['won'])
prediction_clean = prediction_data.dropna(subset=['won'])

# OPTIONAL: Drop rows that are missing too many features (e.g. more than 2 missing)
training_clean = training_clean.dropna(thresh=len(model_features) - 2)
prediction_clean = prediction_clean.dropna(thresh=len(model_features) - 2)




print(f"\nCleaned data shapes:")
print(f"Training: {training_clean.shape}")
print(f"Prediction: {prediction_clean.shape}")

# === Prepare for Neural Network ===
X_train = training_clean[model_features].values
y_train = training_clean['won'].values

X_pred = prediction_clean[model_features].values
y_pred_true = prediction_clean['won'].values

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_pred_scaled = scaler.transform(X_pred)

# Split for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_split, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_pred_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32)


# === Neural Network ===
class WinPredictor(nn.Module):
    def __init__(self, input_dim):
        super(WinPredictor, self).__init__()
        if input_dim < 10:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.model(x)


# Initialize and train
model = WinPredictor(len(model_features))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n=== Training Neural Network ===")
print(f"Input features: {len(model_features)}")
print(f"Training samples: {len(X_train_tensor)}")

# Training loop
epochs = 100
for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation every 20 epochs
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_predictions = (val_outputs > 0.5).float()
            val_accuracy = accuracy_score(y_val_tensor.numpy(), val_predictions.numpy())
            print(
                f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.3f}")

# === Final Predictions ===
model.eval()
with torch.no_grad():
    pred_outputs = model(X_pred_tensor)
    pred_probabilities = pred_outputs.numpy().flatten()
    pred_binary = (pred_outputs > 0.5).float().numpy().flatten()

    actual_wins = y_pred_true.sum()
    predicted_wins = pred_binary.sum()
    accuracy = accuracy_score(y_pred_true, pred_binary)

    print(f"\n=== RESULTS ===")
    print(f"Team {test_team_ids} Analysis:")
    print(f"Total games: {len(pred_probabilities)}")
    print(f"Actual wins: {actual_wins}")
    print(f"Predicted wins: {predicted_wins}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Average win probability: {pred_probabilities.mean():.3f}")



print(f"\n=== CSV COLUMNS USED AS FEATURES ===")
for i, col in enumerate(feature_columns, 1):
    print(f"{i:2d}. {col}")

print("traing colums:",training_clean[feature_columns].describe())
print("fetatrue columns",prediction_clean[feature_columns].describe())