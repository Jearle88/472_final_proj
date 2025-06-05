
import pandas as pd
import numpy as np
from collections import Counter
from math import log2
from sklearn.model_selection import train_test_split
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
df = merged[features + ["win"]].dropna()

# Discretization with safe fallback
def discretize_columns_safe(df, cols, bins=3):
    for col in cols:
        try:
            df[col] = pd.qcut(df[col], q=bins, labels=["low", "medium", "high"], duplicates='drop')
        except ValueError:
            df[col] = "unknown"
    return df

df = discretize_columns_safe(df.copy(), features)

# Split into training and test sets
X = df[features]
y = df["win"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
train_df = X_train.copy()
train_df["win"] = y_train

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

def id3(data, features, target):
    labels = data[target]
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    if len(features) == 0:
        return labels.mode()[0]

    gains = [info_gain(data, attr, target) for attr in features]
    best_attr = features[np.argmax(gains)]
    tree = {best_attr: {}}

    for val in data[best_attr].unique():
        subset = data[data[best_attr] == val]
        if subset.empty:
            tree[best_attr][val] = labels.mode()[0]
        else:
            subtree = id3(subset, [f for f in features if f != best_attr], target)
            tree[best_attr][val] = subtree

    return tree

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = instance[attr]
    if value in tree[attr]:
        return predict(tree[attr][value], instance)
    else:
        return None

# Train and evaluate
tree = id3(train_df, features, "win")
y_pred = [predict(tree, row) for _, row in X_test.iterrows()]
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=["Yes", "No"])

# Output
print("Decision Tree:")
print(tree)
print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
