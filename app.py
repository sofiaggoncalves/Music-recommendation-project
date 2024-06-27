# %%
import requests
import json
from flask import Flask, request, render_template
import re
import nltk
import contractions
from textblob import TextBlob
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import openai
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
import os
import pandas as pd


# %%
#Initialize Flask App
app = Flask(__name__)

# Set the keys
openai.api_key = 'your-key-here' # openAI key
embedding_model = "text-embedding-ada-002"
api_key = 'your-key-here' # Musixmatch API key
spotify_client_id = 'your-client-id-here' # Spotify API credentials
spotify_client_secret = 'your-client-id-here'

# Load the datasets
dataset = pd.read_csv(r"C:\Users\aanas\Desktop\IronWork\GitHub\Music-recommendation-project\datasets\cleaned_dataset_with_lyrics.csv") # Dataset with songs and moods
model_path = r"C:\Users\aanas\Desktop\IronWork\GitHub\Music-recommendation-project\models\best_model" # Trained model path 

# Define the label_to_id dictionary
label_to_id = {'angry': 0, 'happy': 1, 'relaxed': 2, 'sad': 3}

# %%
# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2Model.from_pretrained(model_path)
state_dict = torch.load(os.path.join(model_path, 'classifier_head.pth'), map_location=torch.device('cpu'))
classifier_head = nn.Linear(model.config.n_embd, len(label_to_id))
classifier_head.load_state_dict(state_dict)
model.to('cpu')  # Change to 'cuda' to use GPU
classifier_head.to('cpu')  # Change to 'cuda' to use GPU

# %%

def get_lyrics(song_name, artist_name, api_key):
    base_url = "https://api.musixmatch.com/ws/1.1/"
    search_url = f"{base_url}track.search"
    search_params = {
        'q_track': song_name,
        'q_artist': artist_name,
        'apikey': api_key,
        'f_has_lyrics': 1
    }
    response = requests.get(search_url, params=search_params)
    data = response.json()
    if data['message']['header']['status_code'] != 200:
        return None
    track_list = data['message']['body']['track_list']
    if not track_list:
        return None
    track_id = track_list[0]['track']['track_id']
    lyrics_url = f"{base_url}track.lyrics.get"
    lyrics_params = {
        'track_id': track_id,
        'apikey': api_key
    }
    response = requests.get(lyrics_url, params=lyrics_params)
    data = response.json()
    if data['message']['header']['status_code'] != 200:
        return None
    lyrics = data['message']['body']['lyrics']['lyrics_body']
    return lyrics

def get_song_info_by_lyrics(lyrics, api_key):
    base_url = "https://api.musixmatch.com/ws/1.1/"
    search_url = f"{base_url}track.search"
    search_params = {
        'q_lyrics': lyrics,
        'apikey': api_key,
        'f_has_lyrics': 1
    }
    response = requests.get(search_url, params=search_params)
    data = response.json()
    if data['message']['header']['status_code'] != 200:
        return None
    track_list = data['message']['body']['track_list']
    if not track_list:
        return None
    track = track_list[0]['track']
    song_name = track['track_name']
    artist_name = track['artist_name']
    return song_name, artist_name

# %%
#Preprocessing Lyrics
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

def remove_unwanted_text(lyrics):
    pattern = r'(\*{7} This Lyrics is NOT for Commercial use \*{7}\n\(\d+\))$'
    return re.sub(pattern, '', lyrics).strip()

def eng_ratio(text):
    english_vocab = set(w.lower() for w in words.words())
    text_vocab = set(w.lower() for w in text.split() if w.lower().isalpha())
    if not text_vocab:
        return 0
    unusual = text_vocab.difference(english_vocab)
    return (len(text_vocab) - len(unusual)) / len(text_vocab)

def expand_contractions(text):
    return contractions.fix(text)

def correct_spelling(text):
    return str(TextBlob(text).correct())

def preprocess_lyrics(text):
    text = remove_unwanted_text(text)
    if eng_ratio(text) <= 0.5:
        return None
    text = expand_contractions(text)
    text = correct_spelling(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\n', ' ')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)



# %%
#Embedding
def get_embedding(text, model=embedding_model):
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error: {e}")
        return None

def split_lyrics(lyrics, chunk_size=8191):
    words = lyrics.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


# %%
# Function to predict mood using the trained GPT-2 model
def predict_mood(lyrics):
    # Tokenize the input lyrics
    inputs = tokenizer(lyrics, return_tensors='pt').to('cpu')  # Change to 'cuda' to use GPU

    # Get the model output
    with torch.no_grad():
        outputs = model(**inputs)[0]
        logits = classifier_head(outputs[:, -1, :])

    # Get the predicted mood
    _, predicted_mood = torch.max(logits, dim=1)
    predicted_mood = predicted_mood.item()

    # Convert the predicted mood index to the corresponding mood label
    mood_labels = ['sad', 'happy', 'angry', 'relaxed'] 
    mood = mood_labels[predicted_mood]

    return mood

# %%
# Function to get similar songs from the dataset
def get_similar_songs(predicted_mood, song_name=None, artist_name=None, num_songs=5):
    filtered_songs = dataset[dataset['original_mood'] == predicted_mood]
    if song_name and artist_name:
        filtered_songs = filtered_songs[(filtered_songs['title'].str.lower() != song_name.lower()) & 
                                        (filtered_songs['artist'].str.lower() != artist_name.lower())]
    if len(filtered_songs) < num_songs:
        return filtered_songs[['artist', 'title', 'original_mood']].to_dict(orient='records')
    similar_songs = filtered_songs.sample(n=num_songs)
    return similar_songs[['artist', 'title', 'original_mood']].to_dict(orient='records')


# %%
# Function to get Spotify access token
def get_spotify_token(client_id, client_secret):
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    })
    auth_response_data = auth_response.json()
    return auth_response_data['access_token']

# Function to get Spotify links for songs
def get_spotify_links(songs, access_token):
    headers = {
        'Authorization': f'Bearer {access_token}',
    }
    base_url = 'https://api.spotify.com/v1/search'
    spotify_links = []
    for song in songs:
        query = f"{song['title']} artist:{song['artist']}"
        search_params = {
            'q': query,
            'type': 'track',
            'limit': 1,
        }
        response = requests.get(base_url, headers=headers, params=search_params)
        data = response.json()
        if data['tracks']['items']:
            track = data['tracks']['items'][0]
            spotify_links.append({
                'artist': song['artist'],
                'title': song['title'],
                'spotify_url': track['external_urls']['spotify'],
            })
    return spotify_links

# %%
# Endpoint to process lyrics and predict mood
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_mood', methods=['POST'])
def mood_prediction():
    song_name = request.form['song_name']
    artist_name = request.form['artist_name']
    if not song_name or not artist_name:
        return render_template('index.html', error='Both song name and artist name are required')
    try:
        lyrics = get_lyrics(song_name, artist_name, api_key)
        if not lyrics:
            return render_template('index.html', error='Lyrics not found')
        preprocessed_lyrics = preprocess_lyrics(lyrics)
        predicted_mood = predict_mood(preprocessed_lyrics)
        similar_songs = get_similar_songs(predicted_mood, song_name, artist_name)
        spotify_token = get_spotify_token(spotify_client_id, spotify_client_secret)
        spotify_links = get_spotify_links(similar_songs, spotify_token)
        return render_template('result.html', mood=predicted_mood, songs=spotify_links, 
                               searched_song={'title': song_name, 'artist': artist_name, 'mood': predicted_mood})
    except Exception as e:
        print(f"Error during mood prediction: {e}")
        return render_template('index.html', error=str(e))

@app.route('/search_by_mood', methods=['POST'])
def search_by_mood():
    mood = request.form['mood']
    if not mood:
        return render_template('index.html', error='Mood is required')
    try:
        similar_songs = get_similar_songs(mood)
        spotify_token = get_spotify_token(spotify_client_id, spotify_client_secret)
        spotify_links = get_spotify_links(similar_songs, spotify_token)
        return render_template('result.html', mood=mood, songs=spotify_links)
    except Exception as e:
        print(f"Error during mood search: {e}")
        return render_template('index.html', error=str(e))

@app.route('/search_by_lyrics', methods=['POST'])
def search_by_lyrics():
    lyrics = request.form['lyrics']
    if not lyrics:
        return render_template('index.html', error='Lyrics are required')
    try:
        song_info = get_song_info_by_lyrics(lyrics, api_key)
        if not song_info:
            return render_template('index.html', error='Song not found for the provided lyrics')
        song_name, artist_name = song_info
        lyrics = get_lyrics(song_name, artist_name, api_key)
        preprocessed_lyrics = preprocess_lyrics(lyrics)
        predicted_mood = predict_mood(preprocessed_lyrics)
        similar_songs = get_similar_songs(predicted_mood, song_name, artist_name)
        spotify_token = get_spotify_token(spotify_client_id, spotify_client_secret)
        spotify_links = get_spotify_links(similar_songs, spotify_token)
        return render_template('result.html', mood=predicted_mood, songs=spotify_links, 
                               searched_song={'title': song_name, 'artist': artist_name, 'mood': predicted_mood})
    except Exception as e:
        print(f"Error during lyrics search: {e}")
        return render_template('index.html', error=str(e))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

# %%



