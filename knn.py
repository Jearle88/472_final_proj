import pandas as pd
import numpy as np
from collections import Counter
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from Neural_network_472_final import create_team_records, create_team_features, calculate_opponent_strength

k = 17

def distance(match1, match2):
    return (match1 - match2)**2

def knn(matches, traindata, k):
    retlist = []
    maxdist = -1
    maxindex = 0
    predlist = []
    for matchscore in matches['opponent_strength'].values:
        data = traindata["opponent_strength"].values
        won = traindata["won"].values
        i = 0
        for datascore in data:
            datawon = won[i]
            i += 1
            dist = distance(datascore, matchscore)
            if len(retlist) < k:
                ret = (datawon, dist)
                retlist.append(ret)
                if maxdist == -1 or dist < maxdist:
                    maxdist = dist
                    maxindex = len(retlist) - 1
            else:
                if dist < maxdist:
                    retlist[maxindex] = (datawon, dist)
                    maxdist = dist
                    for i in range(len(retlist)):
                        if retlist[i][1] > maxdist:
                            maxindex = i
                            maxdist = retlist[i][1]
        wincount = 0
        for i in range(len(retlist)):
            wincount += retlist[i][1]
        if wincount > k//2:
            predlist.append(1)
        else:
            predlist.append(0)
    return predlist

csv_path = "all_teams_consolidated_data.csv"
training_team_ids = [382, 360, 103, 256, 360]
prediction_team_id = 83

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

required_cols = ['date', 'homeTeamId', 'awayTeamId']
score_cols = [col for col in df.columns if 'score' in col.lower()]
winner_cols = [col for col in df.columns if 'winner' in col.lower()]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

exclude_from_features = ['homeTeamId', 'awayTeamId', 'date'] + [col for col in df.columns if 'id' in col.lower()]
potential_feature_cols = [col for col in numeric_cols if col not in exclude_from_features]
print(f"Potential feature columns: {potential_feature_cols}")

print("Creating training data...")
training_data = create_team_records(df, training_team_ids)

if len(training_data) == 0:
    print("No training data found!")
    exit()

training_data = calculate_opponent_strength(df, training_data)

training_data = training_data.dropna()

base_features = ['is_home']
score_features = ['lag_teamScore', 'lag_opponentScore', 'lag_won', 'lag_score_diff',
                  'rolling_teamScore_3', 'rolling_opponentScore_3', 'rolling_win_rate_3',
                  'rolling_teamScore_5', 'rolling_win_rate_5', 'rolling_score_diff_5',
                  'recent_form_3', 'recent_form_5', 'home_win_rate']

if 'opponent_strength' in training_data.columns:
    score_features.append('opponent_strength')

additional_features = [col for col in training_data.columns if
                       (col.startswith('rolling_') or col.startswith('lag_')) and
                       col not in score_features]
all_potential_features = base_features + score_features + additional_features
available_features = [f for f in all_potential_features if f in training_data.columns]

prediction_data = create_team_records(df, [prediction_team_id])

if len(prediction_data) == 0:
    print("No prediction data found!")
    exit()

prediction_data = calculate_opponent_strength(df, prediction_data)
prediction_data = prediction_data.dropna()

missing_features = [f for f in available_features if f not in prediction_data.columns]
if missing_features:
    print(f"Warning: Missing features in prediction data: {missing_features}")
    # Remove missing features from the feature list
    available_features = [f for f in available_features if f in prediction_data.columns]
    print(f"Updated available features: {available_features}")


prediction_wins = knn(prediction_data, training_data, k)
total_games = len(prediction_wins)
count = 0
acc_won = prediction_data['won'].values
for i in range(total_games):
    if prediction_wins[i] == acc_won[i]:
        count += 1
print(count, total_games)
accuracy = count / total_games
print(f"Accuracy: {accuracy}")
# predicted_wins_count = sum(prediction_wins)
# predicted_losses_count = total_games - predicted_wins_count
# actual_wins_count = prediction_data['won'].sum()
# actual_losses_count = total_games - actual_wins_count

