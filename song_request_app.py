from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

app = Flask(__name__)

# Load the dataset
file_path = Path('Resources/spotify_dataset.csv')
songs_df = pd.read_csv(file_path)

# Remove duplicate songs
songs_df = songs_df.drop_duplicates(subset=['track', 'artist'], keep='first')

print(f"DataFrame after removing duplicates: {songs_df.shape}")

def convert_uri_to_url(uri):
    if isinstance(uri, str) and uri.startswith("spotify:track:"):
        track_id = uri.split(":")[2]
        return f"https://open.spotify.com/track/{track_id}"
    return None  # Return None for invalid URIs

songs_df['url'] = songs_df['uri'].apply(convert_uri_to_url)

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
 'instrumentalness',
#  'liveness',
#  'valence',
 'tempo',
#  'duration_ms',
#  'time_signature',
#  'chorus_hit',
#  'sections',
 'popularity',
#  'decade'
 ]
X = songs_df[features]

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train a k-Nearest Neighbors model
model = NearestNeighbors(n_neighbors=15, algorithm='ball_tree') 
model.fit(X_scaled)


def recommend_songs(song_title, artist_name, num_recommendations=5):
    # Your existing recommendation code here
    # Make sure to return recommendations instead of printing them
    recommendations = []
    song_row = songs_df[(songs_df['track'] == song_title) & (songs_df['artist'] == artist_name)]
    
    if song_row.empty:
        return None

    song_index = song_row.index[0]
    song_features = X_scaled[song_index].reshape(1, -1)
    distances, indices = model.kneighbors(song_features)
    
    for i in indices.flatten():
        if songs_df.loc[i, 'artist'] != artist_name:
            recommendations.append((songs_df.loc[i, 'track'], songs_df.loc[i, 'artist'], songs_df.loc[i, 'url']))
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