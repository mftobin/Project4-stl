# Project4-stl

# Collaborators: Maura Tobin, Jordan Johnson-Williams, Evan Robinson, Monica Phatangare, Paul Keller

This project used a dataset of 40,000 songs collected from the music streaming app Spotify. This data was
used to create two machine learning models. One was a K Nearest Neighbors model that predicted the most similar songs based on the selection of any given song in the dataset. Another was a Supervised Learning model that tested the reliability of correctly matchings songs in the dataset to its correct genre.

## K Nearest Neighbor
The K Nearest Neigbor model was used to determine what eight songs in the Spotify dataset were most similar to a song selected by the user given the selected song was already in the dataset.

### Dataset
The corresponding dataset contained twenty features:
* Track: Track/Song Title (Object)
* Artist: Performer (Object)
* URI: Spotify Song Identifier (Object)
* Danceability: Dance to Song Scale; 0 = Least Danceable; 1 = Most Danceable (Float64)
* Energy: Energy to Song Scale; 0 = Least Energy; 1 = Most Energy (Float64)
* Key: Key of Song; 0=C, 1=C#/Db, 2=D, 3=D#/Eb, 4=E, 5=F, 6=F#/Gb, 7=G, 8=G#/Ab, 9=A, 10=A#/Bb, 11=B (Int64)
* Loudness: ??? (Float64)
* Mode: Song Charting on Billboard Top 100; 0 = Did not Chart; 1 = Did Chart (Int64)
* Speechiness: Speech to Song Scale; 0 = Least Speech; 1 = Most Speech (Float64)
* Acousticness: Acoustic to Song Scale; 0 = Least Acoustic; 1 = Most Acoustic; (Float64)
* Instrumentalness: Instrument to Song Scale; 0 = Least Instrumental, 1 = Most Instrumental; (Float64)
* Liveness:
* Valence
* Tempo
* Duration_ms
* Time_Signature
* Chorus_Hit
* Sections
* Popularity
* Decades: Decade of Song (Object)

### Procedure
The dataset was cleaned so that no song duplicates existed. Dummy variables for the categorical feature decades were created. In the end the features that were included in the model were Danceability, Energy, Acousticness, Liveness, Valence, Popularity, and each Decade. After applying weights to each of the features, a K Nearest Neighbors model was created. The model was driven by user input. After a song was entered by the user, provided the song is in the dataset, the eight most similar songs determined by the model were produced. Additionally, an app driven by Flask provided a link to the song on Spotify for the user to listen to.