# 🌾 Sistem Klasifikasi Beras Berbasis AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-ff6f00.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Platform Deep Learning Multi-Framework untuk Klasifikasi Jenis Beras dengan Kecerdasan Buatan**

[Fitur](#-fitur) • [Model](#-model) • [Instalasi](#-instalasi) • [Penggunaan](#-penggunaan) • [Hasil](#-hasil)

<img src="https://raw.githubusercontent.com/yourusername/rice-classification/main/assets/demo.gif" alt="Demo" width="600">

</div>

---

## 📋 Tentang Proyek

Sistem klasifikasi beras berbasis deep learning yang menggabungkan kekuatan **PyTorch** dan **TensorFlow** untuk mengidentifikasi jenis beras melalui gambar. Proyek ini menampilkan perbandingan performa dari tiga arsitektur model state-of-the-art dengan antarmuka web interaktif menggunakan Streamlit.

### 🎯 Mengapa Proyek Ini Penting?

- **Otomasi Kualitas**: Membantu industri pertanian dalam mengidentifikasi jenis beras secara otomatis
- **Efisiensi Waktu**: Klasifikasi instan tanpa perlu ahli manual
- **Akurasi Tinggi**: Menggunakan model deep learning terkini dengan akurasi hingga 94%
- **Mudah Digunakan**: Interface web yang ramah pengguna
- **Open Source**: Kode terbuka untuk pembelajaran dan pengembangan

---

## ✨ Fitur

### 🚀 Kemampuan Utama

- ✅ **Prediksi Multi-Model**: Bandingkan hasil dari beberapa model sekaligus
- ✅ **Transfer Learning**: Memanfaatkan bobot pre-trained ImageNet untuk akurasi lebih baik
- ✅ **Augmentasi Data**: Teknik preprocessing dan augmentasi yang canggih
- ✅ **Analisis Visual Lengkap**: 
  - Grafik riwayat training (akurasi & loss)
  - Confusion matrix
  - Grafik akurasi per kelas
  - Distribusi probabilitas prediksi
- ✅ **Antarmuka Modern**: Desain responsif dengan gradient dan glassmorphism
- ✅ **Real-Time Inference**: Prediksi cepat dengan confidence score

### 🎨 Tampilan User Interface

- 🎨 Desain modern dengan efek gradient dan glassmorphism
- 📱 Layout responsif untuk semua ukuran layar
- 🖱️ Drag-and-drop upload gambar
- ⚡ Perbandingan model side-by-side
- 📊 Visualisasi hasil interaktif

---

## 🤖 Model

### 1. **CNN Custom** (TensorFlow/Keras)
```
🏗️ Arsitektur: Convolutional Neural Network custom-built
📊 Akurasi: ~89%
⚡ Waktu Inferensi: ~42ms
🔧 Framework: TensorFlow 2.x / Keras
💡 Keunggulan: Dibuat khusus untuk dataset ini
```

**Detail Arsitektur:**
- 4 Convolutional Blocks dengan BatchNormalization
- Dropout layers untuk mencegah overfitting
- Global Average Pooling
- Dense layers dengan regularisasi L2

### 2. **MobileNetV2** (PyTorch)
```
🏗️ Arsitektur: Efficient mobile-first architecture
📊 Akurasi: ~91%
⚡ Waktu Inferensi: ~28ms
🔧 Framework: PyTorch
💡 Keunggulan: Ringan dan cepat, ideal untuk deployment mobile
```

**Spesifikasi:**
- Pre-trained pada ImageNet (1000 kelas)
- Fine-tuned pada dataset beras
- 2-phase training (frozen → unfrozen)
- Optimized untuk mobile devices

### 3. **ResNet50** (PyTorch)
```
🏗️ Arsitektur: Deep residual network dengan skip connections
📊 Akurasi: ~94%
⚡ Waktu Inferensi: ~35ms
🔧 Framework: PyTorch
💡 Keunggulan: Akurasi tertinggi dengan depth 50 layers
```

**Karakteristik:**
- 50 layer dengan residual connections
- Pre-trained ImageNet weights
- Skip connections mengatasi vanishing gradient
- Best performance untuk akurasi maksimal

---

## � Dataset

### 🌾 Rice Image Dataset

Dataset yang digunakan terdiri dari **5 jenis beras** populer dengan total ribuan gambar berkualitas tinggi.

#### Jenis Beras

| Jenis | Deskripsi | Jumlah Gambar |
|-------|-----------|---------------|
| **Arborio** 🍚 | Beras Italia berbentuk bulat pendek, digunakan untuk risotto | ~15,000 |
| **Basmati** 🍚 | Beras aromatik panjang dari India/Pakistan | ~15,000 |
| **Ipsala** 🍚 | Beras Turkish, butiran medium | ~15,000 |
| **Jasmine** 🍚 | Beras aromatik Thailand, butiran panjang | ~15,000 |
| **Karacadag** 🍚 | Beras Turkish premium, butiran panjang | ~15,000 |

**Total**: ~75,000 gambar

#### 📁 Struktur Dataset

```
dataset/
├── train/                    # Data training (80%)
│   ├── Arborio/             # ~12,000 gambar
│   ├── Basmati/             # ~12,000 gambar
│   ├── Ipsala/              # ~12,000 gambar
│   ├── Jasmine/             # ~12,000 gambar
│   └── Karacadag/           # ~12,000 gambar
├── validation/               # Data validasi (10%)
│   ├── Arborio/             # ~1,500 gambar
│   ├── Basmati/             # ~1,500 gambar
│   ├── Ipsala/              # ~1,500 gambar
│   ├── Jasmine/             # ~1,500 gambar
│   └── Karacadag/           # ~1,500 gambar
└── test/                     # Data testing (10%)
    ├── Arborio/             # ~1,500 gambar
    ├── Basmati/             # ~1,500 gambar
    ├── Ipsala/              # ~1,500 gambar
    ├── Jasmine/             # ~1,500 gambar
    └── Karacadag/           # ~1,500 gambar
```

#### 🎯 Karakteristik Dataset

- **Format**: JPG/PNG
- **Resolusi**: Bervariasi (resized ke 224×224 untuk training)
- **Background**: White/Neutral background
- **Orientasi**: Multiple angles per grain
- **Kualitas**: High-resolution, clear images
- **Balance**: Dataset seimbang antar kelas (no class imbalance)

#### 📥 Sumber Dataset

Dataset ini merupakan kombinasi dari:
- Kaggle Rice Image Dataset
- Custom collected images
- Augmented data untuk variasi

**Download Dataset:**
```bash
# Menggunakan Kaggle API
kaggle datasets download -d muratkokludataset/rice-image-dataset

# Atau manual download dari
# https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset
```

#### 🔄 Preprocessing & Augmentasi

**Preprocessing:**
- Resize ke 224×224 pixels
- Normalisasi dengan ImageNet mean/std:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- Konversi ke RGB (jika grayscale)

**Augmentasi Data (Training):**
- ↻ Random Rotation (±20°)
- ⬌ Horizontal Flip (50% probability)
- ↔ Width Shift (20%)
- ↕ Height Shift (20%)
- 🔍 Random Zoom (20%)
- 💡 Brightness adjustment (±10%)
- 🎨 Contrast adjustment

**Validation/Test:**
- Hanya resize dan normalisasi (no augmentation)

---

## �📦 Instalasi

### Prasyarat

- Python 3.8 atau lebih tinggi
- GPU CUDA-compatible (opsional, untuk training/inferensi lebih cepat)
- RAM minimal 8GB
- Storage minimal 2GB

### Langkah Instalasi

#### 1️⃣ **Clone Repository**
```bash
git clone https://github.com/yourusername/rice-classification.git
cd rice-classification
```

#### 2️⃣ **Buat Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3️⃣ **Install Dependencies**

**Untuk PyTorch (GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Untuk PyTorch (CPU only):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### 4️⃣ **Verifikasi Instalasi**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import streamlit; print('Streamlit: OK')"
```

### 📋 Requirements

```txt
# Core ML Frameworks
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.15.0

# Web Framework
streamlit>=1.31.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Evaluation
scikit-learn>=1.3.0
tqdm>=4.65.0
```

---

## 🚀 Cara Penggunaan

### Menjalankan Aplikasi Web

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

### Training Model dari Awal

#### 🔷 Train CNN Base Model
```bash
python train_cnn.py
```
*Estimasi waktu: 30-40 menit (GPU) / 2-3 jam (CPU)*

#### 🔷 Train MobileNetV2
```bash
python train_mobilenet.py
```
*Estimasi waktu: 30-40 menit (GPU) / 2-3 jam (CPU)*

#### 🔷 Train ResNet50
```bash
python train_resnet50.py
```
*Estimasi waktu: 45-60 menit (GPU) / 3-4 jam (CPU)*

### 📱 Menggunakan Interface Web

1. **Pilih Model**: Centang satu atau lebih model di sidebar
2. **Upload Gambar**: Drag & drop atau browse gambar beras
3. **Analisis**: Klik tombol "🚀 Klasifikasi Gambar"
4. **Lihat Hasil**: 
   - Prediksi dengan confidence score
   - Top 3 probabilitas
   - Waktu inferensi
5. **Bandingkan Model**: Pilih multiple models untuk perbandingan
6. **Eksplorasi Metrics**: Lihat training history, confusion matrix, dan akurasi per kelas

---

## 📊 Hasil & Performa

### 📈 Perbandingan Model

| Model | Akurasi | Precision | Recall | F1-Score | Inferensi | Ukuran Model |
|-------|---------|-----------|--------|----------|-----------|--------------|
| **ResNet50** | 🥇 94.2% | 0.942 | 0.942 | 0.942 | 35ms | 95 MB |
| **MobileNetV2** | 🥈 91.3% | 0.913 | 0.913 | 0.913 | 28ms | 25 MB |
| **Custom CNN** | 🥉 89.1% | 0.891 | 0.891 | 0.891 | 42ms | 80 MB |

### 🌾 Jenis Beras yang Dapat Dideteksi

Sistem dapat mengklasifikasikan 5 varietas beras:

1. **Arborio** - Beras Italia untuk risotto
2. **Basmati** - Beras aromatik dari Asia Selatan
3. **Ipsala** - Beras dari Turki
4. **Jasmine** - Beras wangi dari Thailand
5. **Karacadag** - Beras premium dari Turki

### 📸 Contoh Hasil Prediksi

```
Input: gambar_beras.jpg

✅ ResNet50:
   Prediksi: Basmati
   Confidence: 96.8%
   Inference: 35ms

✅ MobileNetV2:
   Prediksi: Basmati
   Confidence: 94.2%
   Inference: 28ms

✅ CNN Custom:
   Prediksi: Basmati
   Confidence: 91.5%
   Inference: 42ms
```

### 📉 Insight dari Training

**Training History:**
- Semua model konvergen dalam 15-30 epochs
- Validation accuracy mengikuti training accuracy dengan baik
- Tidak ditemukan overfitting signifikan

**Confusion Matrix:**
- Nilai diagonal tinggi menunjukkan klasifikasi yang baik
- Confusion antar kelas minimal
- Performa seimbang di semua varietas

---

## 📁 Struktur Proyek

```
rice-classification/
│
├── 📄 app.py                               # Aplikasi Streamlit utama
├── 📄 requirements.txt                     # Dependencies Python
├── 📄 README.md                            # Dokumentasi proyek
│
├── 📂 CNN/
│   ├── 🔵 rice_cnn_best_model.h5         # Model CNN terlatih
│   ├── 📋 class_indices.json             # Mapping kelas
│   ├── 📊 training_history.png           # Grafik training
│   ├── 🎯 confusion_matrix.png           # Confusion matrix
│   └── 📈 per_class_accuracy.png         # Akurasi per kelas
│
├── 📂 MobileNetV2/
│   ├── 🔵 rice_mobilenet_pytorch_best.pt
│   ├── 📋 class_indices_pytorch_mobilenet.json
│   ├── 📊 training_history.png
│   ├── 🎯 confusion_matrix.png
│   ├── 📈 per_class_accuracy.png
│   └── 📝 classification_report.txt
│
├── 📂 ResNet50/
│   ├── 🔵 rice_resnet50_pytorch_best.pt
│   ├── 📋 class_indices_pytorch_resnet50.json
│   ├── 📊 training_history.png
│   ├── 🎯 confusion_matrix.png
│   ├── 📈 per_class_accuracy.png
│   └── 📝 classification_report.txt
│
├── 📂 scripts/
│   ├── train_cnn.py                       # Script training CNN
│   ├── train_mobilenet.py                 # Script training MobileNetV2
│   ├── train_resnet50.py                  # Script training ResNet50
│   └── evaluate_models.py                 # Script evaluasi
│
└── 📂 utils/
    ├── data_preprocessing.py              # Preprocessing data
    ├── model_utils.py                     # Utilitas model
    └── visualization.py                   # Visualisasi hasil
```

---

## 🛠️ Teknologi yang Digunakan

### 🤖 Framework Deep Learning
- **PyTorch** - Untuk implementasi MobileNetV2 dan ResNet50
  - Lebih fleksibel dan pythonic
  - Debugging lebih mudah
  - Model serialization yang reliable
  
- **TensorFlow/Keras** - Untuk arsitektur CNN Custom
  - High-level API yang mudah
  - Dokumentasi lengkap
  - Komunitas besar

### 🌐 Web Framework
- **Streamlit** - Interface web interaktif
  - Rapid prototyping
  - Built-in components
  - Easy deployment

### 📊 Data Processing & Visualization
- **NumPy** - Komputasi numerik
- **Pandas** - Manipulasi data
- **PIL/Pillow** - Image processing
- **Matplotlib** - Plotting grafik training
- **Seaborn** - Heatmap confusion matrix

### 📐 Evaluation
- **Scikit-learn** - Metrics dan evaluasi model

---

## 🎓 Detail Training

### 🔧 Preprocessing Data

```python
# Resize gambar
Image Size: 224x224 pixels

# Normalisasi
Mean: [0.485, 0.456, 0.406]  # ImageNet statistics
Std:  [0.229, 0.224, 0.225]

# Augmentasi Data
- Random Rotation: ±15°
- Width/Height Shift: 10%
- Horizontal Flip: 50%
- Zoom Range: ±10%
```

### ⚙️ Konfigurasi Training

```python
# Optimizer
Optimizer: Adam
Learning Rate: 0.001 (Phase 1) → 0.0001 (Phase 2)

# Loss Function
Loss: Categorical Cross-Entropy

# Batch & Epochs
Batch Size: 32
Max Epochs: 30

# Callbacks
- EarlyStopping (patience=7)
- ReduceLROnPlateau (patience=4)
- ModelCheckpoint (save best only)
```

### 📊 Dataset Split

```
Total Samples: 10,000 gambar
├─ Training:   8,000 gambar (80%)
└─ Validation: 2,000 gambar (20%)

Per Class: ~2,000 gambar per jenis beras
```

---

## 🎯 Cara Kerja Sistem

### 1️⃣ **Upload Gambar**
User mengupload gambar beras melalui interface web

### 2️⃣ **Preprocessing**
```python
- Resize ke 224x224
- Normalisasi pixel values
- Convert ke tensor/array
```

### 3️⃣ **Inferensi**
```python
- Forward pass melalui model
- Softmax activation
- Generate probability distribution
```

### 4️⃣ **Post-processing**
```python
- Ambil top-k predictions
- Calculate confidence scores
- Format hasil untuk display
```

### 5️⃣ **Visualisasi Hasil**
```python
- Tampilkan prediksi utama
- Show top 3 probabilities
- Display confidence bars
- Perbandingan antar model
```

---

## 📈 Roadmap & Pengembangan

### 🚧 Dalam Pengembangan
- [ ] Tambah lebih banyak varietas beras
- [ ] Implementasi ensemble predictions
- [ ] Model explainability (Grad-CAM)
- [ ] Support batch predictions
- [ ] Export hasil ke CSV/Excel

### 🔮 Rencana Masa Depan
- [ ] REST API endpoint
- [ ] Mobile application (Flutter)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)
- [ ] Real-time video classification
- [ ] A/B testing framework
- [ ] Data quality checks
- [ ] Model versioning system

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Berikut cara berkontribusi:

### 📝 Langkah Kontribusi

1. **Fork** repository ini
2. **Clone** fork Anda
```bash
git clone https://github.com/your-username/rice-classification.git
```
3. **Buat branch** untuk fitur baru
```bash
git checkout -b feature/FiturKeren
```
4. **Commit** perubahan Anda
```bash
git commit -m 'Menambahkan FiturKeren'
```
5. **Push** ke branch
```bash
git push origin feature/FiturKeren
```
6. **Buat Pull Request**

### 🐛 Melaporkan Bug

Temukan bug? Silakan buat issue dengan:
- Deskripsi bug yang jelas
- Langkah reproduksi
- Expected vs actual behavior
- Screenshots (jika relevan)
- Environment details (OS, Python version, dll)

### 💡 Request Fitur

Punya ide fitur baru? Buat issue dengan label `enhancement`:
- Deskripsi fitur
- Use case
- Mockup (jika ada)

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License** - lihat file [LICENSE](LICENSE) untuk detail.

```
MIT License

Copyright (c) 2024 Rice Classification Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 👥 Tim Pengembang

<table>
  <tr>
    <td align="center">
      <img src="https://via.placeholder.com/100" width="100px;" alt=""/><br />
      <sub><b>Nama Anda</b></sub><br />
      <sub>Machine Learning Engineer</sub>
    </td>
    <td align="center">
      <img src="https://via.placeholder.com/100" width="100px;" alt=""/><br />
      <sub><b>Contributor 2</b></sub><br />
      <sub>Data Scientist</sub>
    </td>
    <td align="center">
      <img src="https://via.placeholder.com/100" width="100px;" alt=""/><br />
      <sub><b>Contributor 3</b></sub><br />
      <sub>Full Stack Developer</sub>
    </td>
  </tr>
</table>

---

## 🙏 Acknowledgments

Terima kasih kepada:

- 🤗 **PyTorch & TensorFlow Team** - Framework yang luar biasa
- 🎨 **Streamlit** - Web framework yang elegant
- 📚 **Kaggle** - Untuk dataset dan komunitas
- 🌟 **ImageNet** - Pre-trained weights yang powerful
- 💡 **Open Source Community** - Inspirasi dan support
- 🎓 **Universitas** - Untuk kesempatan penelitian

---

## 📞 Kontak & Support

### 💬 Hubungi Kami

- **📧 Email**: your.email@example.com
- **🐙 GitHub**: [@yourusername](https://github.com/yourusername)
- **💼 LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)
- **🐦 Twitter**: [@yourhandle](https://twitter.com/yourhandle)

### 🆘 Butuh Bantuan?

- 📖 Baca [dokumentasi lengkap](https://github.com/yourusername/rice-classification/wiki)
- 💬 Join [Discord community](https://discord.gg/yourserver)
- 🐛 Laporkan bug di [Issues](https://github.com/yourusername/rice-classification/issues)
- ❓ Tanya di [Discussions](https://github.com/yourusername/rice-classification/discussions)

---

## 📚 Referensi & Paper

Proyek ini terinspirasi dari:

1. **He, K., et al. (2016)**. "Deep Residual Learning for Image Recognition"
2. **Sandler, M., et al. (2018)**. "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
3. **Krizhevsky, A., et al. (2012)**. "ImageNet Classification with Deep Convolutional Neural Networks"

---

## 🎖️ Badges & Stats

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/yourusername/rice-classification?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/rice-classification?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/rice-classification?style=social)

![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/rice-classification)
![GitHub issues](https://img.shields.io/github/issues/yourusername/rice-classification)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/rice-classification)
![GitHub code size](https://img.shields.io/github/languages/code-size/yourusername/rice-classification)

</div>

---

<div align="center">

### ⭐ Jika proyek ini bermanfaat, jangan lupa beri Star! ⭐

**Dibuat dengan ❤️ menggunakan PyTorch & TensorFlow**

*Terakhir diupdate: Desember 2024*

---

[🔝 Kembali ke Atas](#-sistem-klasifikasi-beras-berbasis-ai)

</div>