# JavanE-BIRD: Endemic Bird Sound Detection with Hybrid Gated Fusion

**JavanE-BIRD** is a deep learning-based system designed to detect and classify the vocalizations of endemic bird species in the tropical forests of Java. It utilizes a novel **Hybrid Gated Fusion Architecture** that dynamically weighs features from a CNN (for spectral patterns) and a Transformer (for temporal sequences) to achieve robust performance even in noisy environments.

## 🦅 Key Features

* **Hybrid Architecture**: Combines **ResNet50** (spatial/visual) and a **Lightweight Transformer** (temporal/sequential).
* **Gated Fusion Mechanism**: Implements a learnable "Gate Network" that intelligently decides whether to prioritize visual features (spectrogram) or temporal features (MFCC/Chroma/Tonnetz) for each specific audio sample.
* **Robust Audio Processing**: Handles long-duration recordings by slicing them into 5-second overlapping chunks.
* **Silence/Noise Filtering**: Automatically discards silent segments using RMS thresholding.
* **Interactive UI**: Built with **Streamlit** for real-time inference and result visualization.

## 🧠 Technical Architecture

The model (`hybrid_model.py`) processes audio in two parallel branches:

1.  **Visual Branch (CNN)**
    * **Input**: Log Mel Spectrogram ($128 \times T$).
    * **Backbone**: ResNet50 (pretrained on ImageNet), modified to accept 1-channel (grayscale) input.
    * **Role**: Extracts spatial patterns and texture from the frequency domain.

2.  **Sequential Branch (Transformer)**
    * **Input**: Concatenation of MFCC, Chroma, and Tonnetz features (38 features per timestep).
    * **Backbone**: Custom Transformer Encoder (2 heads, 1 layer).
    * **Role**: Captures temporal dependencies and rhythmic patterns in the bird calls.

3.  **Gated Fusion (The Novelty)**
    Instead of simple concatenation, the model calculates a scalar weight $\alpha$ (0 to 1) via a Sigmoid gate.
    $$Feature_{final} = \alpha \cdot Feature_{CNN} + (1 - \alpha) \cdot Feature_{Transformer}$$
    This allows the model to adaptively "trust" one branch more than the other based on signal quality.

## 🎯 Supported Species

The model is trained to detect the following Javan endemic species:

1.  *Arborophila javanica* (Puyuh-gonggong Jawa)
2.  *Centropus nigrorufus* (Bubut Jawa)
3.  *Cochoa azurea* (Cochoa Jawa)
4.  *Halcyon cyanoventris* (Cekakak Jawa)
5.  *Nisaetus bartelsi* (Elang Jawa)
6.  *Psilopogon javensis* (Takur Tulung-tumpuk)

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/JavanE-BIRD.git](https://github.com/yourusername/JavanE-BIRD.git)
    cd JavanE-BIRD
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate on Windows:
    venv\Scripts\activate
    # Activate on macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have `ffmpeg` installed on your system for `librosa` / `audioread` to work correctly.*

## 🚀 Usage

1.  **Prepare the Model:**
    Ensure your trained model file (`best_model_hybrid.pth`) is located in the root directory of the project.

2.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

3.  **Inference:**
    * Open your browser (usually `http://localhost:8501`).
    * Upload a `.wav` or `.mp3` file (e.g., a forest recording).
    * Adjust the **Confidence Threshold** and **Silence Threshold** in the sidebar.
    * Click **"Analisis Audio"** to view detections.

## 📂 Project Structure

```text
├── app.py                # Main Streamlit application entry point
├── hybrid_model.py       # PyTorch model architecture (CNN + Transformer)
├── best_model_hybrid.pth # Pre-trained model weights (required)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
