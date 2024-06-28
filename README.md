# Mood Classification in Song Lyrics

## Project Overview

This project analyzes song lyrics using natural language processing (NLP) techniques to predict the mood (happy, angry, sad, relaxed) of songs. By leveraging advanced machine learning models, this project explores the intersection of music and data science, providing valuable insights and applications in music recommendation systems and sentiment analysis.

## Problem Statement and Business Case

The objective is to build a model that can predict the mood of a song based on its lyrics. This capability has practical applications in:
- Music recommendation systems for personalized playlists.
- Sentiment analysis in media content.
- Understanding emotional trends in music over time.

## Data Collection and Preprocessing

### Data Sources
- **MoodyLyrics4Q**: Contains 2000 songs categorized into four moods.
- **Dataset-AllMusic-771Lyrics**: Contains 771 songs categorized into the same moods.

### Preprocessing Steps
1. Standardized column names and mood values across datasets.
2. Removed duplicates and NaN values.
3. Obtained lyrics from Musixmatch API.
4. Cleaned and processed lyrics (removed disclaimers, handled contractions, corrected spelling, etc.).
5. Converted mood labels to numerical values using LabelEncoder.

## Exploratory Data Analysis (EDA)
- Analyzed mood distribution, word count, and average word length.
- Identified common bigrams and trigrams.
- Generated bag-of-words representations for each mood.

## Text Preprocessing and Embedding
- Removed stopwords, punctuation, and performed lemmatization.
- Used OpenAI API to obtain embeddings for lyrics.
- Applied PCA for dimensionality reduction of embeddings.

## Machine Learning Model Selection and Evaluation

### Models Used
1. **Support Vector Machine (SVM)**: Baseline model.
2. **BERT**: For contextual understanding of lyrics.
3. **GPT-2**: For generative capabilities and mood prediction.

### Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

### Results
- **SVM**: Accuracy - 0.593
- **BERT**: Accuracy - 0.810
- **GPT-2**: Accuracy - 0.966

GPT-2 was identified as the best-performing model.

## Deployment
- Developed a Flask app that predicts the mood of a song based on its lyrics.
- Users can input a song title and artist, lyrics, or mood to get predictions and recommendations.

## Conclusion and Future Work
This project successfully demonstrated mood prediction in song lyrics using NLP and machine learning. Future work includes refining models, expanding datasets, enhancing text preprocessing, and optimizing the deployment for real-time prediction.

## Instructions on How to Run the Code and Reproduce the Results

### Prerequisites
- Python 3.7+
- Required libraries: pandas, numpy, requests, scikit-learn, nltk, contractions, textblob, wordcloud, openai, tqdm, matplotlib, flask, torch, transformers, joblib, optuna

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/sofiaggoncalves/Music-recommendation-project.git
    ```
2. Navigate to the project directory:
    ```bash
    cd music-recommendation-project
    ```
3. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
4. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
1. Guarantee you have the keys to access all the API's (Spotify, OpenAI and MusixMatch).
2. Run the .py for data cleaning and manipulation, that will generate 3 csv files: dataset-cleaned, and 2 csv with embeddings.
Don't forget to setup the path for the datasets on your directory and add the API keys.
    ```bash
    python data-cleaning-manipulation-process-etl.py
    ```
3. Run the .py for data analysis (optional).
Don't forget to setup the path for the datasets (the cleaned_dataset_with_lyrics just created) on your directory.
    ```bash
    python data-analysis-merged-datasets.py
    ```
3. Run the .py to train the models, that will generate files for each trained model, needed for the app to run.
Don't forget to setup the path where you want to save the models and the path for the datasets (the csv files witthe embeddings just created) on your directory.
    ```bash
    python models-training-and-selection.py
    ```
4. Run the Flask app:
Don't forget to add the API keys and setup the path for the dataset created before (cleaned_data_set_with_lyrics) and the path for the best_model on your directory. The directory for the best model should have all the generated files for that model. Also, guarantee that the templates directory are available in same folder as the app.py
    ```bash
    python app.py
    ```
2. Open a web browser and input song title, artist, or lyrics to get mood predictions and recommendations.

## Assumptions
- The datasets used for training are representative of the various moods.
- The Musixmatch API reliably provides the correct lyrics for the given song titles and artists.
- The preprocessing steps (e.g., removing disclaimers, handling contractions) do not introduce significant errors in the data.
- The embeddings generated by OpenAI's API accurately capture the semantic meaning of the lyrics.
- The selected machine learning models are appropriate for the task and are optimized correctly.

## Tableau Dashboard:
[Metrics Dashboard](https://public.tableau.com/views/IH-final-project/Dashboard1?:language=en-US&publish=yes&:sid=&:display_count=n&:origin=viz_share_link)