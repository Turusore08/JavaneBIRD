import torch
import torch.nn as nn
import torchvision.models as models

class HybridBirdModel(nn.Module):
    def __init__(self, num_classes=6, mode='hybrid'):
        """
        Args:
            num_classes (int): Jumlah spesies burung (target output)
            mode (str): 'hybrid', 'cnn_only', atau 'transformer_only'
        """
        super(HybridBirdModel, self).__init__()
        self.mode = mode
        
        # ==========================================
        # CABANG 1: CNN (Visual / Spasial)
        # ==========================================
        # Menggunakan ResNet50 Pre-trained
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modifikasi layer pertama untuk 1 channel (Grayscale Spectrogram)
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        
        # Encoder CNN (Tanpa FC layer akhir)
        self.cnn_encoder = nn.Sequential(*list(resnet.children())[:-2]) 
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_dim = 2048
        
        # ==========================================
        # CABANG 2: TRANSFORMER (Temporal / Urutan)
        # ==========================================
        # Konfigurasi "Diet Ketat" (Lightweight)
        self.raw_input_dim = 38
        self.trans_embed_dim = 64
        self.trans_heads = 2       # Hanya 2 heads
        self.trans_layers = 1      # Hanya 1 layer
        self.trans_hidden = 64
        
        # Proyeksi Input
        self.input_projection = nn.Linear(self.raw_input_dim, self.trans_embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.trans_embed_dim, 
            nhead=self.trans_heads, 
            dim_feedforward=self.trans_hidden, 
            dropout=0.3,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.trans_layers)
        self.trans_pool = nn.AdaptiveAvgPool1d(1)
        
        # ==========================================
        # GATED FUSION MECHANISM (NOVELTY UTAMA)
        # ==========================================
        # Kita proyeksikan kedua fitur ke dimensi yang sama (512)
        self.common_dim = 512
        
        self.cnn_projector = nn.Sequential(
            nn.Linear(self.cnn_dim, self.common_dim),
            nn.BatchNorm1d(self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.trans_projector = nn.Sequential(
            nn.Linear(self.trans_embed_dim, self.common_dim),
            nn.BatchNorm1d(self.common_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Gate Network: Memutuskan bobot (0-1) untuk CNN vs Transformer
        # Input: Gabungan fitur (512 + 512 = 1024) -> Output: 1 Bobot Scalar
        self.gate_network = nn.Sequential(
            nn.Linear(self.common_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Output 0 sampai 1
        )
        
        # Classifier Akhir (Menerima input 512 hasil fusi)
        self.classifier = nn.Linear(self.common_dim, num_classes)

    def forward(self, img, seq):
        # --- 1. Ekstraksi Fitur CNN ---
        feat_cnn = torch.zeros((img.size(0), self.common_dim), device=img.device)
        
        if self.mode in ['hybrid', 'cnn_only']:
            x = self.cnn_encoder(img)          # (Batch, 2048, H, W)
            x = self.cnn_pool(x).flatten(1)    # (Batch, 2048)
            feat_cnn = self.cnn_projector(x)   # (Batch, 512)
        
        # --- 2. Ekstraksi Fitur Transformer ---
        feat_trans = torch.zeros((seq.size(0), self.common_dim), device=seq.device)
        
        if self.mode in ['hybrid', 'transformer_only']:
            x = self.input_projection(seq)     # (Batch, Time, 64)
            x = self.transformer_encoder(x)
            x = x.permute(0, 2, 1)             # (Batch, 64, Time)
            x = self.trans_pool(x).flatten(1)  # (Batch, 64)
            feat_trans = self.trans_projector(x) # (Batch, 512)
            
        # --- 3. Gated Fusion ---
        if self.mode == 'hybrid':
            # Gabungkan kedua fitur untuk dianalisis oleh Gate
            concat_feats = torch.cat((feat_cnn, feat_trans), dim=1) # (Batch, 1024)
            
            # Hitung bobot gate (alpha)
            # alpha mendekati 1: Percaya CNN
            # alpha mendekati 0: Percaya Transformer
            alpha = self.gate_network(concat_feats)
            
            # Weighted Sum
            combined = (alpha * feat_cnn) + ((1 - alpha) * feat_trans)
            
        elif self.mode == 'cnn_only':
            combined = feat_cnn
        elif self.mode == 'transformer_only':
            combined = feat_trans
            
        # --- 4. Klasifikasi ---
        output = self.classifier(combined)
        return output

if __name__ == "__main__":
    # Sanity Check
    print("🛠️ Testing Gated Hybrid Architecture...")
    model = HybridBirdModel(num_classes=6, mode='hybrid')
    
    dummy_img = torch.randn(4, 1, 128, 313)
    dummy_seq = torch.randn(4, 313, 38)
    
    out = model(dummy_img, dummy_seq)
    print(f"✅ Output Shape: {out.shape} (Harus [4, 6])")
    print("🎉 Arsitektur Gated Fusion Siap!")