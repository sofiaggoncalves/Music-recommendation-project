import requests
import json

api_key = 'b76a1b9a47064853d3363d46e5352e79'

# Function to get lyrics from Musixmatch
def get_lyrics(song_name, artist_name, api_key):
    base_url = "https://api.musixmatch.com/ws/1.1/"
    
    #Endpoint to search for the track
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
        raise Exception("API request failed")
    
    track_list = data['message']['body']['track_list']
    if not track_list:
        return None
    
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
        raise Exception("API request failed")
    
    lyrics = data['message']['body']['lyrics']['lyrics_body']
    return lyrics

# Example usage
api_key = 'b76a1b9a47064853d3363d46e5352e79'
song_name = "Shape of You"
artist_name = "Ed Sheeran"
lyrics = get_lyrics(song_name, artist_name, api_key)
print(lyrics)