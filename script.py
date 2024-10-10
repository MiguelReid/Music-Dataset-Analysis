import pandas as pd
import matplotlib.pyplot as plt
import string
from collections import Counter
import numpy as np

df = pd.read_csv('tcc_ceds_music.csv')

"""
artist_name: The name of the performer or band.
track_name: The title of the song.
release_date: The date the song was released.
genre: The genre classification of the song (e.g., Pop, Rock, Jazz).
lyrics: The song's lyrics (possibly in text form or numerical encoding).
len: The length of the song (probably in seconds or minutes).
dating: A feature indicating mentions of dating or relationships in the song.
violence: Indicates the presence of violent themes in the lyrics.
world/life: Mentions of world events or life reflections in the lyrics.
night/time: Mentions of nighttime or time-related themes.
shake the audience: Suggests whether the song is energetic or engaging.
family/gospel: Mentions of family or religious/gospel themes.
romantic: Indicates the presence of romantic themes.
communication: Mentions of communication (e.g., talking, messaging).
obscene: Indicates the presence of obscene or explicit language in the lyrics.
music: Refers to mentions of music or musical elements in the lyrics.
movement/places: Refers to mentions of movement (e.g., travel) or locations.
light/visual perceptions: Mentions of light or visual-related themes.
family/spiritual: Religious or spiritual mentions related to family.
like/girls: Mentions of attraction toward girls.
sadness: Indicates the presence of sad or melancholic themes in the song.
feelings: Mentions of emotions or feelings in the lyrics.
danceability: A measure (usually 0-1) indicating how suitable the song is for dancing.
loudness: The overall loudness of the track, typically measured in decibels (dB).
acousticness: A measure (0-1) indicating how acoustic the song is.
instrumentalness: A measure (0-1) indicating the absence of vocals in the track.
valence: A measure (0-1) indicating the positivity or happiness conveyed by the track.
energy: A measure (0-1) indicating the intensity or activity level of the track.
topic: The central topic or theme of the song (may be a categorization or description).
age: The age or time period since the song was released. Oldest song from 1950 is 1
"""

# We can observe there's 0 values with null values and 0 duplicates
# print(df.isnull().sum())
# print('Duplicated values -> ', df.duplicated().sum())


######## MOST COMMON TRACK_NAMES #########
duplicated_tracks = df[df.duplicated(subset='track_name', keep=False)]
# Group by track_name and count occurrences
track_name_counts = duplicated_tracks['track_name'].value_counts().head(10)
# print(track_name_counts)
######## MOST COMMON TRACK_NAMES #########


######## MOST COMMON WORDS #########
# Combine all lyrics into a single string
all_lyrics = ' '.join(df['lyrics'].dropna().astype(str))
# Preprocess the text
translator = str.maketrans('', '', string.punctuation)
all_lyrics = all_lyrics.translate(translator).lower().split()
# Count word frequencies
word_counts = Counter(all_lyrics)
# Find the 10 most common words
most_common_words = word_counts.most_common(10)
# print(most_common_words)
######## MOST COMMON WORDS #########


#### MOST COMMON TOPIC PER GENRE #####
# Group by genre and topic, and count occurrences
genre_topic_counts = df.groupby(['genre', 'topic']).size().reset_index(name='counts')

# Identify the most common topic per genre
most_common_topics = genre_topic_counts.loc[genre_topic_counts.groupby('genre')['counts'].idxmax()]

# Show joint labels
labels = most_common_topics.apply(lambda row: f"{row['genre']}: {row['topic']}", axis=1)

# Plot pie chart
plt.figure(figsize=(10, 8))
plt.pie(most_common_topics['counts'], labels=labels, autopct='%1.1f%%')
plt.title('Most Common Topic per Genre')
plt.show()
#### MOST COMMON TOPIC PER GENRE #####


#### SIMILAR SONG FINDER ####
def find_most_similar_song(track_name):
    if track_name not in df['track_name'].values:
        raise ValueError("The input song is not in the DataFrame")

    # Extract the input song details
    input_song = df[df['track_name'] == track_name].iloc[0]

    # Extract the genre and topic of the input song
    genre = input_song['genre']
    topic = input_song['topic']

    # Filter the DataFrame by the same genre and topic excluding input song
    filtered_df = df[(df['genre'] == genre) & (df['topic'] == topic) & (df['track_name'] != track_name)]

    # Define the fields to compare
    fields = [
        'dating', 'violence', 'night/time', 'shake the audience', 'family/gospel',
        'romantic', 'obscene', 'movement/places', 'family/spiritual', 'like/girls',
        'sadness', 'feelings', 'danceability', 'loudness', 'acousticness',
        'instrumentalness', 'valence', 'energy'
    ]

    # Calculate the similarity score
    def similarity_score(row):
        score = 0
        for field in fields:
            score += np.abs(row[field] - input_song[field])
        return score

    # Calculate similarity scores without modifying the DataFrame
    similarity_scores = filtered_df.apply(similarity_score, axis=1)

    # Find the index of the song with the lowest similarity score
    most_similar_index = similarity_scores.idxmin()

    # Return the most similar song
    most_similar_song = filtered_df.loc[most_similar_index]

    return {
        'track_name': most_similar_song['track_name'],
        'artist_name': most_similar_song['artist_name'],
        'release_date': most_similar_song['release_date'].item()
    }
#### SIMILAR SONG FINDER ####



similar_song = find_most_similar_song(
    "you're now tuning into 66.6 fm with dj rapture (the hottest hour of the evening)")
print(similar_song)
