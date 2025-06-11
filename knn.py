import pandas as pd
import numpy as np
# from collections import Counter
# import math
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# from Neural_network_472_final import create_team_records, create_team_features, calculate_opponent_strength

k = 5


def distance(match1, match2, fixturestats, weights):
    dist = 0
    i = 0
    for stat in fixturestats:
        if match1[stat] != "unknown" or match2[stat] != "unknown" or match1[stat] != 0 or match2[stat] != 0:
            dist += (weights[i]*(match1[stat] - match2[stat]))**2
        i += 1
    return dist
    # hometeamid = -1
    # awayteamid = -1
    # for stat in teamstats:
    #     dist += (match1[stat] - match2[stat])**2

def knn(matches, traindata, k, fixturestats, weights):
    retlist = []
    maxdist = -1
    maxindex = 0
    predlist = []
    for _, match in matches.iterrows():
        for _, data in traindata.iterrows():
            datawon = data['win']
            dist = distance(data, match, fixturestats, weights)
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
                s = 0
                if maxdist == 0:
                    retlist[s] = (datawon, dist)
                    s += 1
                    if s >= k:
                        s = 0
        wincount = 0
        for i in range(len(retlist)):
            if retlist[i][0] == "Yes":
                wincount += 1
        if wincount > k//2:
            predlist.append("Yes")
        else:
            predlist.append("No")
    matches["predwins"] = predlist

'''
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
'''
fixtures = pd.read_csv("archive/base_data/fixtures.csv")
team_stats = pd.read_csv("archive/base_data/teamStats.csv")

# Select and prepare relevant columns
selected_stats = [
    "eventId", "teamId", "possessionPct", "wonCorners", "totalShots",
    "shotsOnTarget", "totalPasses", "passPct"
]
team_stats_filtered = team_stats[selected_stats].copy()

# Rename for joining
home_stats = team_stats_filtered.rename(columns=lambda x: f"home_{x}" if x != "eventId" else x)
away_stats = team_stats_filtered.rename(columns=lambda x: f"away_{x}" if x != "eventId" else x)

# Merge with fixture data
merged = fixtures.merge(home_stats, left_on=["eventId", "homeTeamId"], right_on=["eventId", "home_teamId"], how="inner")
merged = merged.merge(away_stats, left_on=["eventId", "awayTeamId"], right_on=["eventId", "away_teamId"], how="inner")
merged["win"] = merged["homeTeamWinner"].apply(lambda x: "Yes" if x else "No")

# Final feature set
features = [
    "home_possessionPct", "home_wonCorners", "home_totalShots",
    "home_shotsOnTarget", "home_totalPasses", "home_passPct"
]
weights = [.01, .01, .03, .03, .001, 1]
df = merged[features + ["win", "home_teamId"]].dropna()

def discretize_columns_safe(df, cols, bins=3):
    for col in cols:
        try:
            df[col] = pd.qcut(df[col], q=bins, labels=[1, 2, 3], duplicates='drop')
        except ValueError:
            df[col] = "unknown"
    return df

# df = discretize_columns_safe(df.copy(), features)

unique_teams = sorted(df["home_teamId"].unique())
n = len(unique_teams)
train_ids = set(unique_teams[:int(n * 0.7)])
val_ids = set(unique_teams[int(n * 0.7):int(n * 0.8)])
test_ids = set(unique_teams[int(n * 0.8):])

train_df = df[df["home_teamId"].isin(train_ids)].drop(columns=["home_teamId"])
val_df = df[df["home_teamId"].isin(val_ids)].drop(columns=["home_teamId"])
test_df = df[df["home_teamId"].isin(test_ids)].drop(columns=["home_teamId"])

'''
prediction_data = calculate_opponent_strength(df, prediction_data)
prediction_data = prediction_data.dropna()

missing_features = [f for f in available_features if f not in prediction_data.columns]
if missing_features:
    print(f"Warning: Missing features in prediction data: {missing_features}")
    # Remove missing features from the feature list
    available_features = [f for f in available_features if f in prediction_data.columns]
    print(f"Updated available features: {available_features}")
'''

knn(test_df, train_df, k, features, weights)
total_games = len(test_df["predwins"])
count = 0
for index,row in test_df.iterrows():
    if row["predwins"] == row["win"]:
        count += 1
print(count, total_games)
accuracy = count / total_games
print(f"Accuracy: {accuracy}")
# predicted_wins_count = sum(prediction_wins)
# predicted_losses_count = total_games - predicted_wins_count
# actual_wins_count = prediction_data['won'].sum()
# actual_losses_count = total_games - actual_wins_count

