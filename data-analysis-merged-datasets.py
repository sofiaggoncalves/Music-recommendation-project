# %%
import pandas as pd
import numpy as np
from nltk import ngrams
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re


# Load the dataset
file_path = r"C:\Users\aanas\Desktop\IronWork\GitHub\Music-recommendation-project\datasets\cleaned_dataset_with_lyrics.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
display(df.info())
display(df.describe())
display(df.head())

# Handle NaN values in the lyrics column
df['lyrics'] = df['lyrics'].fillna('')

# Function to clean the lyrics
def clean_lyrics(text):
    text = text.replace("******* This Lyrics is NOT for Commercial use *******", "").strip()
    text = text.replace("(1409624691312)", "").strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Clean the lyrics
df['lyrics'] = df['lyrics'].apply(clean_lyrics)

# Function to count words in a string
def word_count(text):
    words = text.split()
    return len(words)

# Function to calculate average word length in a string
def avg_word_length(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return np.mean([len(word) for word in words])

# Adding new columns to the dataframe
df['word_count'] = df['lyrics'].apply(word_count)
df['avg_word_length'] = df['lyrics'].apply(avg_word_length)

# Save the updated dataframe to a new CSV file
df.to_csv('updated_dataset_with_word_counts.csv', index=False)

# Function to get the most frequent n-grams
def get_top_ngrams(corpus, n, top_k):
    n_grams = ngrams(corpus.split(), n)
    n_grams_freq = Counter(n_grams)
    return n_grams_freq.most_common(top_k)

# Concatenate all lyrics into one large corpus
corpus = ' '.join(df['lyrics'])

# Get the top 10 bigrams and trigrams
top_bigrams = get_top_ngrams(corpus, 2, 10)
top_trigrams = get_top_ngrams(corpus, 3, 10)

# Convert n-grams to DataFrame for visualization
bigrams_df = pd.DataFrame(top_bigrams, columns=['bigram', 'count'])
trigrams_df = pd.DataFrame(top_trigrams, columns=['trigram', 'count'])

# Save n-grams data to CSV
bigrams_df.to_csv('top_bigrams.csv', index=False)
trigrams_df.to_csv('top_trigrams.csv', index=False)

# Plot the top bigrams
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='bigram', data=bigrams_df)
plt.title('Top 10 Bigrams')
plt.xlabel('Count')
plt.ylabel('Bigram')
plt.show()

# Plot the top trigrams
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='trigram', data=trigrams_df)
plt.title('Top 10 Trigrams')
plt.xlabel('Count')
plt.ylabel('Trigram')
plt.show()

# Data Analysis: Visualize the distribution of the moods
sns.countplot(x='original_mood', data=df)
plt.title('Mood Distribution')
plt.show()


