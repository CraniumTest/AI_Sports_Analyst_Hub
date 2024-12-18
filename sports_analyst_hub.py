import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from transformers import pipeline

class GameAnalysis:
    def __init__(self, play_data):
        self.play_data = pd.DataFrame(play_data)
        self.commentary_pipeline = pipeline("text-generation", model="distilgpt2")

    def generate_commentary(self):
        commentary = []
        for _, row in self.play_data.iterrows():
            play_description = f"Team {row['team']} {row['action']} with {row['outcome']}."
            commentary.append(self.commentary_pipeline(play_description, max_length=50)[0]['generated_text'])
        return commentary

class PlayerPerformance:
    def __init__(self, player_data):
        self.player_data = pd.DataFrame(player_data)

    def evaluate_performance(self, player_name):
        player_stats = self.player_data[self.player_data['player'] == player_name]
        return player_stats.describe()

class PredictiveAnalytics:
    def __init__(self, historical_data):
        self.historical_data = pd.DataFrame(historical_data)
        self.model = LinearRegression()

    def train_model(self):
        X = self.historical_data[['feature1', 'feature2', 'feature3']]
        y = self.historical_data['outcome']
        self.model.fit(X, y)

    def predict_outcome(self, feature_data):
        feature_array = np.array(feature_data).reshape(1, -1)
        outcome = self.model.predict(feature_array)
        return outcome[0]

# Example usage
if __name__ == "__main__":
    play_data = [
        {"team": "A", "action": "scored", "outcome": "goal"},
        {"team": "B", "action": "missed shot", "outcome": "no goal"}
    ]
    game_analysis = GameAnalysis(play_data)
    print(game_analysis.generate_commentary())

    player_data = [
        {"player": "John Doe", "game": "Game 1", "points": 22, "assists": 5, "rebounds": 10},
        {"player": "Jane Smith", "game": "Game 1", "points": 30, "assists": 7, "rebounds": 12},
        {"player": "John Doe", "game": "Game 2", "points": 28, "assists": 4, "rebounds": 9}
    ]
    player_performance = PlayerPerformance(player_data)
    print(player_performance.evaluate_performance("John Doe"))

    historical_data = [
        {"feature1": 1, "feature2": 2, "feature3": 3, "outcome": 15},
        {"feature1": 2, "feature2": 4, "feature3": 6, "outcome": 30},
        {"feature1": 3, "feature2": 6, "feature3": 9, "outcome": 45}
    ]
    predictive_analytics = PredictiveAnalytics(historical_data)
    predictive_analytics.train_model()
    print(predictive_analytics.predict_outcome([4, 8, 12]))
