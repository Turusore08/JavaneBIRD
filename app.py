import streamlit as st
import torch
import librosa
import numpy as np
import soundfile as sf
import io
import pandas as pd

# --- KONFIGURASI ---
MODEL_PATH = 'best_model_hybrid.pth' 
SR = 32000
DURATION = 5  
N_MELS = 128
INPUT_LENGTH = SR * DURATION


CLASSES = [
    'Arborophila javanica',   
    'Centropus nigrorufus',  
    'Cochoa azurea',        
    'Halcyon cyanoventris',   
    'Nisaetus bartelsi',   
    'Psilopogon javensis'     
]

import torch.nn as nn


@st.cache_resource
def load_model():
    """Load model sekali saja ke cache memory"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    from hybrid_model import HybridBirdModel 
    
    model = HybridBirdModel(num_classes=len(CLASSES))

    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

def preprocess_audio_chunk(y):
    """Mengubah raw audio 5 detik menjadi Tensor Input Model"""

    if len(y) < INPUT_LENGTH:
        y = np.pad(y, (0, INPUT_LENGTH - len(y)))
    else:
        y = y[:INPUT_LENGTH]


    melspec = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=2048, hop_length=512
    )
    log_mel = librosa.power_to_db(melspec, ref=np.max)

    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    spec_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 128, T)

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, n_chroma=12)
    tonnetz = librosa.feature.tonnetz(y=y, sr=SR)
    
    seq_feat = np.concatenate([mfcc, chroma, tonnetz], axis=0).T # (T, 38)
    
    mean = np.mean(seq_feat, axis=0)
    std = np.std(seq_feat, axis=0) + 1e-6
    seq_feat = (seq_feat - mean) / std
    
    seq_tensor = torch.tensor(seq_feat, dtype=torch.float32).unsqueeze(0) # (1, T, 38)
    
    return spec_tensor, seq_tensor

def is_silent(y, threshold=0.01):
    """Cek apakah audio cuma hening/noise rendah"""
    rms = np.sqrt(np.mean(y**2))
    return rms < threshold

st.set_page_config(page_title="Javan Endemic Bird Detector", page_icon="🐦")

st.title("🦅 Deteksi Burung Endemik Jawa (AI Hybrid)")
st.markdown("""
Sistem ini menggunakan **Hybrid CNN-Transformer** untuk mendeteksi spesies burung dalam rekaman panjang.
Model mampu menangani *noise* hutan dan memberikan bukti potongan suara.
""")

st.sidebar.header("⚙️ Konfigurasi")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.70, help="Hanya tampilkan prediksi dengan keyakinan di atas nilai ini.")
silence_threshold = st.sidebar.slider("Silence Threshold (RMS)", 0.0, 0.1, 0.01, step=0.005, help="Abaikan segmen audio yang volumenya di bawah nilai ini.")
uploaded_file = st.file_uploader("Upload File Audio (WAV/MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("🔍 Analisis Audio"):
        try:
            model, device = load_model()
        except Exception as e:
            st.error(f"Gagal memuat model. Pastikan file '{MODEL_PATH}' dan 'hybrid_model.py' ada.\nError: {e}")
            st.stop()

        with st.spinner('Sedang memproses audio...'):
            y_full, sr_orig = librosa.load(uploaded_file, sr=SR)
            duration_total = len(y_full) / SR
            
            st.info(f"Durasi Audio: {duration_total:.2f} detik. Memecah menjadi segmen {DURATION} detik...")
            
            detections = []

            progress_bar = st.progress(0)
            step = int(SR * DURATION) 
            
            for i in range(0, len(y_full), step):

                progress = min((i / len(y_full)), 1.0)
                progress_bar.progress(progress)
                

                end = i + step
                chunk = y_full[i:end]
                

                if len(chunk) < SR * 1.0: 
                    continue

                if is_silent(chunk, threshold=silence_threshold):
                    continue

                spec, seq = preprocess_audio_chunk(chunk)

                with torch.no_grad():
                    output = model(spec, seq)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    confidence, pred_idx = torch.max(probs, 1)
                    
                confidence = confidence.item()
                label = CLASSES[pred_idx.item()]

                if confidence >= conf_threshold:
                    start_time = i / SR
                    end_time = (i + len(chunk)) / SR
                    
                    detections.append({
                        "Spesies": label,
                        "Waktu": f"{start_time:.1f}s - {end_time:.1f}s",
                        "Confidence": f"{confidence*100:.1f}%",
                        "Audio_Chunk": chunk  
                    })

            progress_bar.progress(1.0)

        st.divider()
        st.subheader("📋 Hasil Deteksi")
        
        if len(detections) == 0:
            st.warning("🚫 Tidak ada burung yang terdeteksi dalam rekaman ini (atau confidence terlalu rendah).")
        else:
            st.success(f"Ditemukan {len(detections)} segmen suara burung!")
            

            for det in detections:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{det['Spesies']}**")
                        st.caption(f"🕒 {det['Waktu']} | 🎯 Keyakinan: {det['Confidence']}")
                    with col2:
                        virtual_file = io.BytesIO()
                        sf.write(virtual_file, det['Audio_Chunk'], SR, format='WAV')
                        st.audio(virtual_file, format='audio/wav')
                    st.divider()