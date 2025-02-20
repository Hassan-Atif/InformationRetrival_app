import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO

st.title('Reuters Article Similarity App')

#Load of embeddings and reuters articles
embeddings = np.load("document_embeddings.npy")
embeddings = embeddings[:500]
articles = np.load("articles.npy", allow_pickle=True)  

st.write(f"Number of Loaded articles are: {len(articles)} ")

# Model selection dropdown
models = ['Model A', 'Model B', 'Model C']
selected_model = st.selectbox('Select a model:', models)

# User text input
user_input = st.text_input('Enter your text:')

# Submit button
if st.button('Submit'):
    if user_input.strip():
        user_embedding = np.random.rand(1, embeddings.shape[1])  # Simulate embedding

        # Compute cosine similarity
        similarities = cosine_similarity(user_embedding, embeddings)

        # Get top-k most similar indexes
        top_k = 5
        top_k_indexes = np.argsort(similarities[0])[-top_k:][::-1]

        # 🔥 Display 50-character snippet instead of index
        st.write("Top most relevant snippets are below:")
        for idx in top_k_indexes:
            snippet = articles[idx][:50]  # Get first 50 characters of the article
            st.write(f"🔹 `{snippet}...`")

    else:
        st.warning("Error: Please enter text before submitting.")
