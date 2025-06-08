import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from collections import defaultdict

# === Configuration ===
csv_path = "C:/Users/Johnt/OneDrive/Documents/GitHub/472_final_proj/all_teams_consolidated_data.csv"
training_team_ids = [382]  # 10 teams for training
prediction_team_id = 83  # New team to predict

# === Load and Prepare Data ===
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# === Dynamically Identify Available Columns ===
print("Available columns in CSV:")
print(df.columns.tolist())

# Identify key columns dynamically
required_cols = ['date', 'homeTeamId', 'awayTeamId']
score_cols = [col for col in df.columns if 'score' in col.lower()]
winner_cols = [col for col in df.columns if 'winner' in col.lower()]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nIdentified score columns: {score_cols}")
print(f"Identified winner columns: {winner_cols}")
print(f"Numeric columns available: {numeric_cols}")

# Remove ID and date columns from numeric features
exclude_from_features = ['homeTeamId', 'awayTeamId', 'date'] + [col for col in df.columns if 'id' in col.lower()]
potential_feature_cols = [col for col in numeric_cols if col not in exclude_from_features]
print(f"Potential feature columns: {potential_feature_cols}")


def create_team_records(df, team_ids):
    """Create records for multiple teams using dynamic column detection"""
    all_records = []

    # Detect column mappings
    home_score_col = next((col for col in df.columns if 'home' in col.lower() and 'score' in col.lower()), None)
    away_score_col = next((col for col in df.columns if 'away' in col.lower() and 'score' in col.lower()), None)
    home_winner_col = next((col for col in df.columns if 'home' in col.lower() and 'winner' in col.lower()), None)
    away_winner_col = next((col for col in df.columns if 'away' in col.lower() and 'winner' in col.lower()), None)

    print(f"Using columns - Home Score: {home_score_col}, Away Score: {away_score_col}")
    print(f"Using columns - Home Winner: {home_winner_col}, Away Winner: {away_winner_col}")

    # Get all additional numeric columns that could be features
    base_cols = ['date', 'homeTeamId', 'awayTeamId', home_score_col, away_score_col, home_winner_col, away_winner_col]
    additional_cols = [col for col in potential_feature_cols if col not in base_cols]

    for team_id in team_ids:
        team_records = []

        # Get all matches for this team
        team_matches = df[(df['homeTeamId'] == team_id) | (df['awayTeamId'] == team_id)].copy()
        team_matches = team_matches.sort_values('date')

        for _, row in team_matches.iterrows():
            base_record = {
                'team_id': team_id,
                'date': row['date'],
            }

            if row['homeTeamId'] == team_id:
                base_record.update({
                    'is_home': 1,
                    'teamScore': row[home_score_col] if home_score_col else 0,
                    'opponentScore': row[away_score_col] if away_score_col else 0,
                    'opponentId': row['awayTeamId'],
                    'won': int(row[home_winner_col]) if home_winner_col else 0,
                })
            else:  # away team
                base_record.update({
                    'is_home': 0,
                    'teamScore': row[away_score_col] if away_score_col else 0,
                    'opponentScore': row[home_score_col] if home_score_col else 0,
                    'opponentId': row['homeTeamId'],
                    'won': int(row[away_winner_col]) if away_winner_col else 0,
                })

            # Add all additional numeric columns as features
            for col in additional_cols:
                if col in row and pd.notna(row[col]):
                    base_record[f'game_{col}'] = row[col]

            team_records.append(base_record)

        # Convert to DataFrame and create features for this team
        df_team = pd.DataFrame(team_records)
        if len(df_team) > 0:
            df_team = create_team_features(df_team, additional_cols)
            all_records.append(df_team)

    # Combine all teams
    return pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()


def create_team_features(df_team, additional_cols):
    """Create lagged and rolling features for a single team with dynamic columns"""
    # Sort by date
    df_team = df_team.sort_values('date').copy()

    # Basic lagged features
    df_team['lag_teamScore'] = df_team['teamScore'].shift(1)
    df_team['lag_opponentScore'] = df_team['opponentScore'].shift(1)
    df_team['lag_won'] = df_team['won'].shift(1)
    df_team['lag_score_diff'] = (df_team['teamScore'] - df_team['opponentScore']).shift(1)

    # Rolling averages (last 3 games)
    df_team['rolling_teamScore_3'] = df_team['teamScore'].rolling(window=3, min_periods=1).mean()
    df_team['rolling_opponentScore_3'] = df_team['opponentScore'].rolling(window=3, min_periods=1).mean()
    df_team['rolling_win_rate_3'] = df_team['won'].rolling(window=3, min_periods=1).mean()

    # Rolling averages (last 5 games)
    df_team['rolling_teamScore_5'] = df_team['teamScore'].rolling(window=5, min_periods=1).mean()
    df_team['rolling_win_rate_5'] = df_team['won'].rolling(window=5, min_periods=1).mean()
    df_team['rolling_score_diff_5'] = (df_team['teamScore'] - df_team['opponentScore']).rolling(window=5,
                                                                                                min_periods=1).mean()

    # Form indicators
    df_team['recent_form_3'] = df_team['won'].rolling(window=3, min_periods=1).sum()  # wins in last 3
    df_team['recent_form_5'] = df_team['won'].rolling(window=5, min_periods=1).sum()  # wins in last 5

    # Home/away performance (rolling)
    df_team['home_win_rate'] = df_team.groupby('is_home')['won'].expanding().mean().reset_index(level=0, drop=True)

    # Create rolling features for additional numeric columns
    for col in additional_cols:
        game_col = f'game_{col}'
        if game_col in df_team.columns:
            # Create rolling averages for the additional stats
            df_team[f'rolling_{col}_3'] = df_team[game_col].rolling(window=3, min_periods=1).mean()
            df_team[f'rolling_{col}_5'] = df_team[game_col].rolling(window=5, min_periods=1).mean()
            df_team[f'lag_{col}'] = df_team[game_col].shift(1)

    return df_team


def calculate_opponent_strength(df, team_records):
    """Calculate opponent strength features"""
    # Calculate each team's overall win rate
    team_win_rates = df.groupby(['homeTeamId']).apply(
        lambda x: (x['homeTeamWinner'].sum() +
                   df[df['awayTeamId'] == x.name]['awayTeamWinner'].sum()) /
                  (len(x) + len(df[df['awayTeamId'] == x.name]))
    ).to_dict()

    # Add away team win rates
    away_win_rates = df.groupby(['awayTeamId']).apply(
        lambda x: (df[df['homeTeamId'] == x.name]['homeTeamWinner'].sum() +
                   x['awayTeamWinner'].sum()) /
                  (len(df[df['homeTeamId'] == x.name]) + len(x))
    ).to_dict()

    # Combine win rates
    for team_id, win_rate in away_win_rates.items():
        if team_id in team_win_rates:
            # Average the two calculations
            team_win_rates[team_id] = (team_win_rates[team_id] + win_rate) / 2
        else:
            team_win_rates[team_id] = win_rate

    # Add opponent strength to team records
    team_records['opponent_strength'] = team_records['opponentId'].map(team_win_rates).fillna(0.5)

    return team_records


# === Create Training Data ===
print("Creating training data...")
training_data = create_team_records(df, training_team_ids)

if len(training_data) == 0:
    print("No training data found!")
    exit()

# Add opponent strength features
training_data = calculate_opponent_strength(df, training_data)

# Drop rows with insufficient data (first few games might have NaN due to rolling windows)
training_data = training_data.dropna()

print(f"Training data shape: {training_data.shape}")
print(f"Teams in training: {sorted(training_data['team_id'].unique())}")

# === Dynamically Determine Available Features ===
# Base features that should always be available
base_features = ['is_home']

# Add score-related features
score_features = ['lag_teamScore', 'lag_opponentScore', 'lag_won', 'lag_score_diff',
                  'rolling_teamScore_3', 'rolling_opponentScore_3', 'rolling_win_rate_3',
                  'rolling_teamScore_5', 'rolling_win_rate_5', 'rolling_score_diff_5',
                  'recent_form_3', 'recent_form_5', 'home_win_rate']

# Add opponent strength if available
if 'opponent_strength' in training_data.columns:
    score_features.append('opponent_strength')

# Find additional rolling/lag features that were created from the CSV columns
additional_features = [col for col in training_data.columns if
                       (col.startswith('rolling_') or col.startswith('lag_')) and
                       col not in score_features]

# Combine all features
all_potential_features = base_features + score_features + additional_features

# Only use features that actually exist in the data
available_features = [f for f in all_potential_features if f in training_data.columns]

print(f"Available features ({len(available_features)}): {available_features}")

# === Create Prediction Data (for the new team) ===
print("Creating prediction data...")
prediction_data = create_team_records(df, [prediction_team_id])

if len(prediction_data) == 0:
    print("No prediction data found!")
    exit()

prediction_data = calculate_opponent_strength(df, prediction_data)
prediction_data = prediction_data.dropna()

print(f"Prediction data shape: {prediction_data.shape}")

# Ensure prediction data has the same features
missing_features = [f for f in available_features if f not in prediction_data.columns]
if missing_features:
    print(f"Warning: Missing features in prediction data: {missing_features}")
    # Remove missing features from the feature list
    available_features = [f for f in available_features if f in prediction_data.columns]
    print(f"Updated available features: {available_features}")

# === Prepare Training Data ===
X_train_full = training_data[available_features].values
y_train_full = training_data['won'].values

# === Prepare Prediction Data ===
X_pred = prediction_data[available_features].values
y_pred_true = prediction_data['won'].values  # True outcomes for evaluation

# === Normalize Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_pred_scaled = scaler.transform(X_pred)

# === Split training data for validation ===
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_pred = torch.tensor(X_pred_scaled, dtype=torch.float32)
y_pred_true = torch.tensor(y_pred_true, dtype=torch.float32).unsqueeze(1)


# === Enhanced Neural Network ===
class EnhancedWinPredictor(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(EnhancedWinPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# === Initialize Model ===
model = EnhancedWinPredictor(input_dim=len(available_features))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# === Training Loop with Early Stopping ===
epochs = 200
best_val_loss = float('inf')
patience = 20
patience_counter = 0

print("\nStarting training...")
for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    train_outputs = model(X_train)
    train_loss = criterion(train_outputs, y_train)
    train_loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

        # Calculate validation accuracy
        val_predictions = (val_outputs > 0.5).float()
        val_accuracy = accuracy_score(y_val.numpy(), val_predictions.numpy())

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.3f}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))


# === Season Prediction Function ===
def predict_season_record(model, scaler, df, team_id, season_start_date=None, season_end_date=None):
    """
    Predict a team's record over a specific season or time period using dynamic features
    """
    # Get all matches for the team
    team_matches = df[(df['homeTeamId'] == team_id) | (df['awayTeamId'] == team_id)].copy()
    team_matches = team_matches.sort_values('date')

    # Filter by season dates if provided
    if season_start_date:
        team_matches = team_matches[team_matches['date'] >= pd.to_datetime(season_start_date)]
    if season_end_date:
        team_matches = team_matches[team_matches['date'] <= pd.to_datetime(season_end_date)]

    if len(team_matches) == 0:
        return None, []

    # Create team records using the same dynamic approach
    prediction_team_data = create_team_records(df, [team_id])

    if len(prediction_team_data) == 0:
        return None, []

    prediction_team_data = calculate_opponent_strength(df, prediction_team_data)

    # Filter by date range if specified
    if season_start_date:
        prediction_team_data = prediction_team_data[prediction_team_data['date'] >= pd.to_datetime(season_start_date)]
    if season_end_date:
        prediction_team_data = prediction_team_data[prediction_team_data['date'] <= pd.to_datetime(season_end_date)]

    # Remove games with insufficient historical data for features
    prediction_team_data = prediction_team_data.dropna()

    if len(prediction_team_data) == 0:
        return None, []

    # Use the same features that were used for training
    missing_features = [f for f in available_features if f not in prediction_team_data.columns]
    if missing_features:
        print(f"Warning: Missing features for prediction: {missing_features}")
        return None, []

    # Prepare features
    X_season = prediction_team_data[available_features].values
    X_season_scaled = scaler.transform(X_season)
    X_season_tensor = torch.tensor(X_season_scaled, dtype=torch.float32)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_season_tensor)
        win_probabilities = predictions.numpy().flatten()
        predicted_wins = (predictions > 0.5).float().numpy().flatten()

    # Calculate season record
    total_games = len(predicted_wins)
    predicted_wins_count = int(predicted_wins.sum())
    predicted_losses_count = total_games - predicted_wins_count
    actual_wins_count = prediction_team_data['won'].sum()
    actual_losses_count = total_games - actual_wins_count

    # Create detailed game-by-game results
    game_results = []
    for i, (_, game) in enumerate(prediction_team_data.iterrows()):
        game_results.append({
            'date': game['date'],
            'is_home': bool(game['is_home']),
            'opponent_id': game['opponentId'],
            'predicted_win': bool(predicted_wins[i]),
            'win_probability': win_probabilities[i],
            'actual_win': bool(game['won']),
            'team_score': game['teamScore'],
            'opponent_score': game['opponentScore']
        })

    season_summary = {
        'team_id': team_id,
        'total_games': total_games,
        'predicted_record': f"{predicted_wins_count}-{predicted_losses_count}",
        'actual_record': f"{actual_wins_count}-{actual_losses_count}",
        'predicted_win_pct': predicted_wins_count / total_games if total_games > 0 else 0,
        'actual_win_pct': actual_wins_count / total_games if total_games > 0 else 0,
        'prediction_accuracy': accuracy_score(prediction_team_data['won'].values, predicted_wins),
        'avg_win_probability': win_probabilities.mean(),
        'season_start': prediction_team_data['date'].min(),
        'season_end': prediction_team_data['date'].max()
    }

    return season_summary, game_results


# === Final Evaluation ===
model.eval()
with torch.no_grad():
    # Validation set performance
    val_outputs = model(X_val)
    val_predictions = (val_outputs > 0.5).float()
    val_accuracy = accuracy_score(y_val.numpy(), val_predictions.numpy())

    print(f"\n=== Model Performance ===")
    print(f"Validation Accuracy (on training teams): {val_accuracy:.3f}")

# === Predict Full Season Record ===
print(f"\n=== Season Record Prediction for Team {prediction_team_id} ===")

# You can specify a season date range, or leave None to use all available data
# Example: predict for 2023 season
# season_summary, game_results = predict_season_record(
#     model, scaler, df, prediction_team_id,
#     season_start_date='2023-01-01',
#     season_end_date='2023-12-31'
# )

# Predict for all available games
season_summary, game_results = predict_season_record(model, scaler, df, prediction_team_id)

if season_summary:
    print(f"Team ID: {season_summary['team_id']}")
    print(
        f"Season Period: {season_summary['season_start'].strftime('%Y-%m-%d')} to {season_summary['season_end'].strftime('%Y-%m-%d')}")
    print(f"Total Games: {season_summary['total_games']}")
    print(f"Predicted Record: {season_summary['predicted_record']} ({season_summary['predicted_win_pct']:.3f} win %)")
    print(f"Actual Record: {season_summary['actual_record']} ({season_summary['actual_win_pct']:.3f} win %)")
    print(f"Prediction Accuracy: {season_summary['prediction_accuracy']:.3f}")
    print(f"Average Win Probability: {season_summary['avg_win_probability']:.3f}")

    # Show confidence levels
    high_confidence_games = [g for g in game_results if g['win_probability'] > 0.7 or g['win_probability'] < 0.3]
    print(f"High Confidence Predictions (>70% or <30%): {len(high_confidence_games)}/{len(game_results)} games")

    # Show some specific game predictions
    print(f"\n=== Sample Game Predictions ===")
    for i, game in enumerate(game_results[:10]):  # Show first 10 games
        location = "Home" if game['is_home'] else "Away"
        prediction = "Win" if game['predicted_win'] else "Loss"
        actual = "Won" if game['actual_win'] else "Lost"
        print(f"Game {i + 1} ({game['date'].strftime('%Y-%m-%d')}): "
              f"{location} vs Team {game['opponent_id']} - "
              f"Predicted: {prediction} ({game['win_probability']:.3f}), "
              f"Actual: {actual} ({game['team_score']}-{game['opponent_score']})")

    # Analyze prediction patterns
    home_games = [g for g in game_results if g['is_home']]
    away_games = [g for g in game_results if not g['is_home']]

    if home_games and away_games:
        home_pred_win_pct = sum(g['predicted_win'] for g in home_games) / len(home_games)
        away_pred_win_pct = sum(g['predicted_win'] for g in away_games) / len(away_games)

        print(f"\n=== Home vs Away Predictions ===")
        print(f"Home Games: {len(home_games)} - Predicted Win %: {home_pred_win_pct:.3f}")
        print(f"Away Games: {len(away_games)} - Predicted Win %: {away_pred_win_pct:.3f}")

else:
    print("No valid season data found for prediction.")


# === Future Season Prediction (if you want to predict unknown outcomes) ===
def predict_future_games(model, scaler, future_fixtures, team_id):
    """
    Predict outcomes for future/scheduled games where you don't know the result yet
    This would work with a fixtures file that includes future games
    """
    # This function would work similarly but for games without known outcomes
    # You'd need to prepare the features based on the team's current form
    pass