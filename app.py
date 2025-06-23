import streamlit as st
import pickle
import os
import gdown
from wordcloud import WordCloud
import matplotlib.pyplot as plt

MODEL_URL = "https://drive.google.com/uc?id=19h_PmH-MdkmKUebcqQiq--6itfodcqcM"
MODEL_PATH = "bertopic_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mengunduh model dari Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

topic_model = load_model()

st.title("🎯 Prediksi Topik dari Komentar TikTok")
st.markdown("Masukkan satu atau beberapa komentar TikTok untuk melihat topik dominan menurut model IndoBERT + BERTopic.")

user_input = st.text_area("📝 Masukkan komentar TikTok (pisahkan dengan baris baru)", height=200)

if st.button("🔍 Analisis Topik"):
    if user_input.strip() == "":
        st.warning("Masukkan setidaknya satu komentar terlebih dahulu.")
    else:
        comments = [line.strip() for line in user_input.split("\n") if line.strip()]
        topics, probs = topic_model.transform(comments)

        st.subheader("📌 Hasil Topik Modeling")
        for i, (comment, topic_id, prob) in enumerate(zip(comments, topics, probs)):
            st.markdown(f"**Komentar {i+1}:** {comment}")
            if topic_id == -1:
                st.warning("Topik tidak dikenali.")
            else:
                st.success(f"Prediksi Topik: `{topic_id}` (Probabilitas: {prob:.2f})")
                words = topic_model.get_topic(topic_id)
                st.markdown("**Keyword Utama:** " + ", ".join([w[0] for w in words]))
                wordcloud = WordCloud(width=600, height=300, background_color='white')
                wc = wordcloud.generate(" ".join([w[0] for w in words]))
                st.image(wc.to_array())
