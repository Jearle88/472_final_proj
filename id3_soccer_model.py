import pandas as pd
import numpy as np
from collections import Counter
from math import log2
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets
fixtures = pd.read_csv("fixtures.csv")
team_stats = pd.read_csv("teamStats.csv")

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
df = merged[features + ["win", "home_teamId"]].dropna()

# Discretization with 4 bins
def discretize_columns_safe(df, cols, bins=4):
    for col in cols:
        try:
            df[col] = pd.qcut(df[col], q=bins, labels=[f"bin{i}" for i in range(bins)], duplicates='drop')
        except ValueError:
            df[col] = "unknown"
    return df

df = discretize_columns_safe(df.copy(), features, bins=4)

# Partitioning teams
unique_teams = sorted(df["home_teamId"].unique())
n = len(unique_teams)
train_ids = set(unique_teams[:int(n * 0.7)])
val_ids = set(unique_teams[int(n * 0.7):int(n * 0.8)])
test_ids = set(unique_teams[int(n * 0.8):])

train_df = df[df["home_teamId"].isin(train_ids)].drop(columns=["home_teamId"])
val_df = df[df["home_teamId"].isin(val_ids)].drop(columns=["home_teamId"])
test_df = df[df["home_teamId"].isin(test_ids)].drop(columns=["home_teamId"])

# ID3 Functions
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * log2(count / total) for count in counts.values())

def info_gain(data, attr, target):
    total_entropy = entropy(data[target])
    values = data[attr].unique()
    weighted_entropy = 0
    for v in values:
        subset = data[data[attr] == v]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy

def id3(data, features, target, depth=0, max_depth=5, min_samples_split=5):
    labels = data[target]
    if len(labels.unique()) == 1 or not features or depth == max_depth or len(data) < min_samples_split:
        return labels.mode()[0]

    gains = [info_gain(data, attr, target) for attr in features]
    best_attr = features[np.argmax(gains)]
    tree = {best_attr: {}}

    for val in data[best_attr].unique():
        subset = data[data[best_attr] == val]
        if subset.empty:
            tree[best_attr][val] = labels.mode()[0]
        else:
            subtree = id3(
                subset,
                [f for f in features if f != best_attr],
                target,
                depth=depth + 1,
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )
            tree[best_attr][val] = subtree

    return tree

def predict(tree, instance, default="Yes"):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = instance.get(attr, None)
    if value in tree[attr]:
        return predict(tree[attr][value], instance, default)
    else:
        return default  # fallback to majority class

# Train and evaluate
majority_class = train_df["win"].mode()[0]
tree = id3(train_df, features, "win", max_depth=5, min_samples_split=5)
y_pred = [predict(tree, row, default=majority_class) for _, row in test_df[features].iterrows()]
accuracy = accuracy_score(test_df["win"], y_pred)
conf_matrix = confusion_matrix(test_df["win"], y_pred, labels=["Yes", "No"])

# Output
print("Decision Tree:")
print(tree)
print("\nAccuracy on test set:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
