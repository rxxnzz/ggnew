import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Fungsi untuk mendapatkan genre lagu berdasarkan link Last.fm
def get_song_genre(lastfm_link):
    # Melakukan permintaan ke halaman Last.fm
    response = requests.get(lastfm_link)
    # Memeriksa status kode permintaan
    if response.status_code == 200:
        # Parsing halaman menggunakan BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        # Mencari elemen dengan class "tags-list" yang berisi genre
        genre_list = soup.find(class_='tags-list')
        # Mengambil genre lagu dari elemen genre_list
        genres = [genre.text for genre in genre_list.find_all('a')]
        return genres
    else:
        return None

# Load data
data = pd.read_csv('2017sd2021.csv')
st.title('Rekomendasi Musik')
selected_year = st.slider("Pilih Tahun", 2017, 2021)
data = data[(data['Years'] == selected_year)]
# Menggabungkan kolom genre menjadi satu kolom 'genres'
data['genres'] = data[['Genre', 'Genre2', 'Genre3', 'Genre4', 'Genre5', 
                       'Genre6', 'Genre7', 'Genre8', 'Genre9', 
                       'Genre10']].apply(lambda x: ' '.join(x.dropna()), axis=1)
# Membangun matriks TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['genres'].values.astype('U'))
# Streamlit app
# Input link Last.fm
if selected_year != 0 :
    lastfm_link = st.text_input('Masukkan link Last.fm untuk lagu')
# Tombol untuk mendapatkan rekomendasi lagu
if st.button('Dapatkan Rekomendasi'):
    # Memastikan link Last.fm diisi sebelum memproses
    if lastfm_link:
        # Memanggil fungsi get_song_genre untuk mendapatkan genre lagu dari link Last.fm
        genres_link = get_song_genre(lastfm_link)
        # Memeriksa apakah genre ditemukan atau tidak
        if genres_link:
            # Menggabungkan genre dari link dengan genre dari file
            combined_genres = genres_link + data['genres'].tolist()
            # Membangun matriks TF-IDF untuk genre yang digabungkan
            tfidf_matrix_combined = tfidf.fit_transform(combined_genres)
            # Menghitung matriks kesamaan kosinus untuk genre yang digabungkan
            cosine_sim_combined = linear_kernel(tfidf_matrix_combined, tfidf_matrix_combined)
            # Mendapatkan indeks lagu yang dipilih
            index = len(genres_link)  # Indeks genre dari link Last.fm
            # Menghitung skor kesamaan kosinus untuk semua lagu
            similarity_scores = list(enumerate(cosine_sim_combined[index]))
            # Mengurutkan lagu berdasarkan skor kesamaan
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            # Mendapatkan 5 lagu yang paling mirip
            top_similar_songs = similarity_scores[1:6]
            # Mendapatkan judul lagu dan artis yang direkomendasikan dari file
            recommended_songs = data.iloc[[i[0] for i in top_similar_songs]][['Track Name', 'Artist']]
            # Menampilkan lagu-lagu yang direkomendasikan
            st.subheader('Lagu-lagu yang Direkomendasikan:')
            for i, song in enumerate(recommended_songs.iterrows()):
                st.write(f"{i+1}. {song[1]['Track Name']} - {song[1]['Artist']}")
        else:
            st.error('Gagal mendapatkan genre dari link Last.fm')
    else:
        st.warning('Masukkan link Last.fm untuk lagu terlebih dahulu')
