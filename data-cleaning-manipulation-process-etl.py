# %%
import pandas as pd
import requests
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
import re
import nltk
import contractions
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import openai
import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


#API Keys
api_key = 'your-key-here' # Musixmatch API key
openai.api_key = 'your-key-here' # OpenAI API key
embedding_model = "text-embedding-ada-002"

# Datasets path
moody_lyrics_path = r"C:\Users\aanas\Desktop\IronWork\GitHub\Music-recommendation-project\datasets\MoodyLyrics4Q.csv"
allmusic_lyrics_path = r"C:\Users\aanas\Desktop\IronWork\GitHub\Music-recommendation-project\datasets\Dataset-AllMusic-771Lyrics.csv"

# Load datasets
moody_lyrics = pd.read_csv(moody_lyrics_path)
allmusic_lyrics = pd.read_csv(allmusic_lyrics_path, encoding='latin1')

# Remove index columns
if 'index' in moody_lyrics.columns:
    moody_lyrics.drop(columns=['index'], inplace=True)
if 'Name' in allmusic_lyrics.columns:
    allmusic_lyrics.drop(columns=['Name'], inplace=True)
print("Index columns removed from datasets.")

# Replace "_" with " " in Dataset-AllMusic-771 and classification replacements
allmusic_lyrics.replace('_', ' ', regex=True, inplace=True)
allmusic_lyrics.replace('  ', ' ', regex=True, inplace=True)
classification_replacements = {'Q1': 'happy', 'Q2': 'angry', 'Q3': 'sad', 'Q4': 'relaxed'}
allmusic_lyrics['Classification'].replace(classification_replacements, inplace=True)
print(f"Classification replacements done: Q1 = happy, Q2 = angry, Q3 = sad, Q4 = relaxed.")

# Standardize column names
moody_lyrics.rename(columns={'title': 'title', 'artist': 'artist', 'mood': 'mood'}, inplace=True)
allmusic_lyrics.rename(columns={'Title': 'title', 'Artist': 'artist', 'Classification': 'mood'}, inplace=True)
print(f"Column names were standardized.")

# Merge datasets vertically
merged_dataset = pd.concat([moody_lyrics, allmusic_lyrics], ignore_index=True)
print("Datasets merged into a unique dataset.")

# Remove rows with NaN values
initial_row_count = merged_dataset.shape[0]
merged_dataset.dropna(inplace=True)
nan_removed_count = initial_row_count - merged_dataset.shape[0]
print(f"Removed {nan_removed_count} rows containing NaN values.")

# Remove duplicate rows
initial_row_count = merged_dataset.shape[0]
merged_dataset.drop_duplicates(inplace=True)
duplicates_removed_count = initial_row_count - merged_dataset.shape[0]
print(f"Removed {duplicates_removed_count} duplicate rows.")

# Keep original categorical data and convert to numerical
merged_dataset['original_mood'] = merged_dataset['mood']
le = LabelEncoder()
merged_dataset['mood'] = le.fit_transform(merged_dataset['mood'])
print("Categorical data converted to numerical.")

# Function to get lyrics from Musixmatch
def get_lyrics(song_name, artist_name, api_keys):
    base_url = "https://api.musixmatch.com/ws/1.1/"
    
    for api_key in api_keys:
        # Endpoint to search for the track
        search_url = f"{base_url}track.search"
        search_params = {
            'q_track': song_name,
            'q_artist': artist_name,
            'apikey': api_key,
            'f_has_lyrics': 1
        }
        
        # Making the request to search for the track
        response = requests.get(search_url, params=search_params)
        data = response.json()
        
        if data['message']['header']['status_code'] != 200:
            continue
        
        track_list = data['message']['body']['track_list']
        if not track_list:
            continue
        
        # Getting the track ID of the first result
        track_id = track_list[0]['track']['track_id']
        
        # Endpoint to get lyrics
        lyrics_url = f"{base_url}track.lyrics.get"
        lyrics_params = {
            'track_id': track_id,
            'apikey': api_key
        }
        
        # Making the request to get the lyrics
        response = requests.get(lyrics_url, params=lyrics_params)
        data = response.json()
        
        if data['message']['header']['status_code'] != 200:
            continue
        
        lyrics = data['message']['body']['lyrics']['lyrics_body']
        return lyrics
    
    return None

# Fetching lyrics for each song
missing_lyrics_count = 0

def fetch_lyrics(row):
    global missing_lyrics_count
    lyrics = get_lyrics(row['title'], row['artist'], api_key)
    if lyrics is None:
        print(f"No lyrics found for '{row['title']}' by {row['artist']}")
        missing_lyrics_count += 1
    return lyrics

merged_dataset['lyrics'] = merged_dataset.apply(fetch_lyrics, axis=1)
initial_row_count = merged_dataset.shape[0]
print("Lyrics fetched and added to the dataset.")
print(f"Total songs in dataset: {initial_row_count}")
print(f"Total songs without lyrics: {missing_lyrics_count}")

# Remove rows without lyrics
initial_row_count = merged_dataset.shape[0]
merged_dataset = merged_dataset[merged_dataset['lyrics'].notna()]
lyrics_removed_count = initial_row_count - merged_dataset.shape[0]
print(f"Removed {lyrics_removed_count} rows without lyrics.")
print(f"Total songs in dataset: {merged_dataset.shape[0]}")

# Data Cleaning: Handle missing values, if any
merged_dataset.dropna(subset=['lyrics', 'mood'], inplace=True)


# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the dataset for preprocessing
df = merged_dataset

# Data Cleaning: Feature Selection
df = df[['lyrics', 'mood']]

# Remove rows with NaN lyrics, if any
df = df.dropna(subset=['lyrics'])

# Remove unwanted text from the end of the lyrics
def remove_unwanted_text(lyrics):
    pattern = r'(\*{7} This Lyrics is NOT for Commercial use \*{7}\n\(\d+\))$'
    return re.sub(pattern, '', lyrics).strip()

df['lyrics'] = df['lyrics'].apply(remove_unwanted_text)

# Function to check if the lyrics are in English
def eng_ratio(text):
    ''' Returns the ratio of English to total words from a text '''
    english_vocab = set(w.lower() for w in words.words())
    text_vocab = set(w.lower() for w in text.split() if w.lower().isalpha())
    if not text_vocab:
        return 0  # Avoid division by zero
    unusual = text_vocab.difference(english_vocab)
    eng_ratio = (len(text_vocab) - len(unusual)) / len(text_vocab)
    return eng_ratio

# Filter out non-English lyrics
print("Filtering non-English lyrics...")
initial_row_count = df.shape[0]
df['eng_ratio'] = df['lyrics'].apply(eng_ratio)
df = df[df['eng_ratio'] > 0.5]
rows_removed_count = initial_row_count - df.shape[0]
print(f"Removed {rows_removed_count} rows containing non-English lyrics.")

# Function to expand contractions
def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

# Function to correct spelling
def correct_spelling(text):
    corrected_text = TextBlob(text).correct()
    return str(corrected_text)

# Text Preprocessing function
def preprocess_lyrics(text):
    text = expand_contractions(text)  # Expand contractions
    text = correct_spelling(text)  # Correct spelling
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = text.replace('\n', ' ')  # Remove line breaks
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(tokens)  # Join back into a single string

# Apply preprocessing to the lyrics
print("Preprocessing lyrics...")
df['lyrics'] = df['lyrics'].apply(preprocess_lyrics)

# Drop the eng_ratio column as it is no longer needed
df = df.drop(columns=['eng_ratio'])

# Function to get embeddings from OpenAI API
def get_embedding(text: str, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(
            input=[text],
            model=model
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to split lyrics into smaller chunks
def split_lyrics(lyrics, chunk_size=8191):
    words = lyrics.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Generate embeddings for the lyrics
print("Generating embeddings for lyrics...")
embeddings = []
for lyrics in tqdm.tqdm(df['lyrics']):
    chunks = split_lyrics(lyrics)
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
    if chunk_embeddings and all(e is not None for e in chunk_embeddings):
        embeddings.append(np.mean(chunk_embeddings, axis=0))
    else:
        embeddings.append(None)

df['embedding'] = embeddings

# Save the embeddings to a new CSV file
df.to_csv('exploratory_data_analysis/lyrics_embeddings.csv', index=False)
print("Embeddings saved to 'exploratory_data_analysis/lyrics_embeddings.csv'")

# Remove rows with None embeddings
df = df.dropna(subset=['embedding'])

# Reduce dimensionality using PCA
print("Reducing dimensionality using PCA...")
embeddings = np.array(df['embedding'].tolist())
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(embeddings)
df['reduced_embedding'] = reduced_embeddings.tolist()

# Save the embeddings with reduced dimensionality to a new CSV file
df.to_csv('exploratory_data_analysis/lyrics_embeddings_reduced.csv', index=False)
print("Reduced embeddings saved to 'exploratory_data_analysis/lyrics_embeddings_reduced.csv'")


