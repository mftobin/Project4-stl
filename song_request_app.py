import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the dataset
file_path = Path('Resources/spotify_dataset.csv')
songs_df = pd.read_csv(file_path)

# Remove duplicate songs
songs_df = songs_df.drop_duplicates(subset=['track', 'artist'], keep='first')

print(f"DataFrame after removing duplicates: {songs_df.shape}")

songs_df = songs_df.reset_index(drop=True)

def convert_uri_to_url(uri):
    if isinstance(uri, str) and uri.startswith("spotify:track:"):
        track_id = uri.split(":")[2]
        return f"https://open.spotify.com/track/{track_id}"
    return None  # Return None for invalid URIs

songs_df['url'] = songs_df['uri'].apply(convert_uri_to_url)

# Normalize dataset columns for searching
songs_df['track_lower'] = songs_df['track'].str.strip().str.lower()
songs_df['artist_lower'] = songs_df['artist'].str.strip().str.lower()

songs_df['decade'] = songs_df['decade'].replace({
    '60s': 1960.0,
    '70s': 1970.0,
    '80s': 1980.0,
    '90s': 1990.0,
    '00s': 2000.0, 
    '10s': 2010.0
}).astype(float)

# Select relevant features for the model

features = [
# 'track',
#  'artist',
#  'uri',
 'danceability',
 'energy',
#  'key',
#  'loudness',
#  'mode',
#  'speechiness',
 'acousticness',
#  'instrumentalness',
#  'liveness',
 'valence',
#  'tempo',
#  'duration_ms',
#  'time_signature',
#  'chorus_hit',
#  'sections',
 'popularity',
 'decade'
 ]


X = songs_df[features]

scaler = MinMaxScaler()
# Ensure indices of scaled features match songs_df
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=features,
    index=songs_df.index  # Match the original index
)

# Feature weighting
weights = {
    'danceability': 1.3,
    'energy': 1.8,
    'acousticness': 0.8,
    'valence': 1.5,
    'popularity': 0.5,
    'decade': 0.4
}

# Apply the weights to the scaled features
weighted_features = X_scaled * pd.Series(weights)

# Train a k-Nearest Neighbors model
model = NearestNeighbors(n_neighbors=15, algorithm='ball_tree') 
model.fit(weighted_features)


def recommend_songs(song_title, artist_name, num_recommendations=5):
    # Make sure to return recommendations instead of printing them
    # Preprocess input
    song_title = song_title.strip().lower()
    artist_name = artist_name.strip().lower()
    
    # Filter the dataset for the input song and artist with case-insensitive partial matching
    song_row = songs_df[
        (songs_df['track_lower'].str.contains(song_title)) & 
        (songs_df['artist_lower'].str.contains(artist_name))
    ]
    
    if song_row.empty:
        print(f"Error: Song '{song_title}' by '{artist_name}' not found in the dataset.")
        return

    # Get the index and weighted features of the input song
    song_index = song_row.index[0]
    song_features = weighted_features.loc[song_index].values.reshape(1, -1)  # Use weighted_features here

    # find nearest neighbors
    distances, indices = model.kneighbors(song_features)

    # Map indices back to the original DataFrame
    recommendations = []
    
    for i in indices.flatten():
        original_index = songs_df.index[i]  # Map to original index
        if (songs_df.loc[original_index, 'track_lower'] != song_row.iloc[0]['track_lower'] and 
            songs_df.loc[original_index, 'artist_lower'] != song_row.iloc[0]['artist_lower']):
            recommendations.append((songs_df.loc[original_index, 'track'], songs_df.loc[original_index, 'artist'], songs_df.loc[original_index, 'url']))
        if len(recommendations) >= num_recommendations:
            break

    return recommendations

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        song_title = request.form['song_title']
        artist_name = request.form['artist_name']
        num_recommendations = int(request.form['num_recommendations'])
        recommendations = recommend_songs(song_title, artist_name, num_recommendations)
    
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)