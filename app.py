import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import time
import os
import tensorflow as tf
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Rice Classification",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS untuk tampilan modern dan clean
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background - Elegant Blue & White */
    .stApp {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 25%, #ffffff 50%, #dbeafe 75%, #eff6ff 100%);
    }
    
    .main {
        background: transparent;
        padding: 2rem 3rem;
    }
    
    .block-container {
        background: transparent;
        max-width: 1400px;
    }
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 40px rgba(37, 99, 235, 0.2);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.85);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Section cards */
    .section-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.1);
        height: 100%;
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #93c5fd;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: #f0f9ff;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: #3b82f6;
        background: #dbeafe;
    }
    
    /* Image preview */
    .image-preview {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.12);
        margin: 1rem 0;
        border: 1px solid #dbeafe;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #60a5fa 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(37, 99, 235, 0.25);
        color: white;
    }
    
    .result-model {
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-bottom: 0.75rem;
    }
    
    .result-class {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .result-confidence {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-time {
        font-size: 0.875rem;
        opacity: 0.8;
    }
    
    /* Probability bars */
    .prob-container {
        background: #f0f9ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.75rem 0;
    }
    
    .prob-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    
    /* Progress bar custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 50%, #60a5fa 100%);
        height: 8px;
        border-radius: 10px;
    }
    
    .stProgress > div > div {
        background: #e0e7ff;
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #60a5fa 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e40af 0%, #2563eb 50%, #3b82f6 100%);
        padding: 1.5rem 1rem;
    }
    
    [data-testid="stSidebar"] h3 {
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #bfdbfe;
        margin-bottom: 1rem;
        margin-top: 1rem;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0f2fe;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(191, 219, 254, 0.2);
        margin: 1.5rem 0;
    }
    
    /* Info box */
    .info-container {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .info-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .info-item:last-child {
        border-bottom: none;
    }
    
    .info-label {
        font-weight: 600;
        color: #bfdbfe;
        font-size: 0.875rem;
    }
    
    .info-value {
        font-weight: 700;
        color: #ffffff;
        font-size: 0.875rem;
    }
    
    /* Model badges */
    .model-list-item {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .model-check {
        color: #86efac;
        font-weight: 700;
    }
    
    .model-name {
        font-weight: 600;
        color: #ffffff;
        font-size: 0.875rem;
    }
    
    .model-framework {
        color: #bfdbfe;
        font-size: 0.75rem;
    }
    
    /* Comparison table */
    .dataframe {
        border: none !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff;
        border-radius: 10px;
        font-weight: 600;
        padding: 1rem;
        border: 1px solid #bfdbfe;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #93c5fd;
        border-radius: 12px;
        padding: 2rem;
        background: #f0f9ff;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: #dbeafe;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
        color: #1e40af !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #64748b !important;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 10px;
        border: 1px solid #bfdbfe;
        background: #eff6ff;
    }
    
    .stAlert [data-testid="stMarkdownContainer"] {
        color: #1e40af !important;
    }
    
    /* Info/Warning/Error messages */
    .stInfo {
        background: #eff6ff !important;
        border-left: 4px solid #3b82f6 !important;
        color: #1e40af !important;
    }
    
    .stWarning {
        background: #fef3c7 !important;
        border-left: 4px solid #f59e0b !important;
        color: #92400e !important;
    }
    
    .stError {
        background: #fee2e2 !important;
        border-left: 4px solid #ef4444 !important;
        color: #991b1b !important;
    }
    
    /* Multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        border-radius: 6px;
        color: white !important;
    }
    
    /* Multiselect dropdown */
    [data-baseweb="popover"] {
        background: #ffffff !important;
    }
    
    [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(147, 197, 253, 0.5) !important;
        border-radius: 8px !important;
        color: #1e40af !important;
    }
    
    [data-baseweb="select"] input {
        color: #1e40af !important;
    }
    
    [data-baseweb="select"] ul {
        background: #ffffff !important;
    }
    
    [data-baseweb="select"] li {
        background: #ffffff !important;
        color: #1e40af !important;
        font-weight: 500;
    }
    
    [data-baseweb="select"] li:hover {
        background: #eff6ff !important;
        color: #2563eb !important;
    }
    
    /* Multiselect in sidebar */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMultiSelect input {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMultiSelect svg {
        fill: white !important;
    }
    
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background: rgba(255, 255, 255, 0.25) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Multiselect placeholder text */
    [data-baseweb="select"] [data-baseweb="input"] {
        color: #1e40af1e40af !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="input"] {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Caption text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }
    
    /* General text color */
    p, span, div {
        color: #1e293b;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🌾 Rice Classification System</div>
    <div class="main-subtitle">AI-Powered Multi-Framework Deep Learning Platform</div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'predictions_results' not in st.session_state:
    st.session_state.predictions_results = []

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load class indices
@st.cache_resource
def load_class_indices():
    possible_paths = [
        'CNN/class_indices.json',
        'Mobile Net V2/class_indices_pytorch_mobilenet.json',
        'RestNet50/class_indices_pytorch_resnet50.json',
        'RestNet50/class_indices_resnet50.json',
        'class_indices.json',
        'class_indices_pytorch_mobilenet.json',
        'class_indices_pytorch_resnet50.json'
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            continue
    
    st.error("❌ Class indices file not found!")
    return None

class_indices = load_class_indices()
if class_indices is None:
    st.stop()

num_classes = len(class_indices)

# Load PyTorch models
@st.cache_resource
def load_pytorch_models():
    models_dict = {}
    
    try:
        mobilenet_path = 'Mobile Net V2/rice_mobilenet_pytorch_best.pt'
        if os.path.exists(mobilenet_path):
            mobilenet = models.mobilenet_v2(pretrained=False)
            mobilenet.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(mobilenet.last_channel, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )
            mobilenet.load_state_dict(torch.load(mobilenet_path, map_location=device))
            mobilenet.to(device)
            mobilenet.eval()
            models_dict['MobileNetV2 (PyTorch)'] = mobilenet
    except:
        pass
    
    try:
        resnet_path = 'RestNet50/rice_resnet50_pytorch_best.pt'
        if os.path.exists(resnet_path):
            resnet = models.resnet50(pretrained=False)
            num_features = resnet.fc.in_features
            resnet.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )
            resnet.load_state_dict(torch.load(resnet_path, map_location=device))
            resnet.to(device)
            resnet.eval()
            models_dict['ResNet50 (PyTorch)'] = resnet
    except:
        pass
    
    return models_dict

@st.cache_resource
def load_keras_models():
    keras_models_dict = {}
    
    try:
        cnn_path = 'CNN/rice_cnn_best_model.h5'
        if os.path.exists(cnn_path):
            class CustomDense(tf.keras.layers.Dense):
                def __init__(self, *args, quantization_config=None, **kwargs):
                    super().__init__(*args, **kwargs)
            
            custom_objects = {'Dense': CustomDense}
            
            try:
                cnn_model = keras.models.load_model(cnn_path, custom_objects=custom_objects, compile=False)
            except:
                cnn_model = tf.keras.models.load_model(cnn_path, compile=False, safe_mode=False)
            
            keras_models_dict['CNN (Keras)'] = cnn_model
    except:
        pass
    
    return keras_models_dict

models_dict = load_pytorch_models()
keras_models_dict = load_keras_models()
models_dict.update(keras_models_dict)

if not models_dict:
    st.error("No models loaded!")
    st.stop()

def preprocess_image_pytorch(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor

def preprocess_image_keras(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_pytorch(model, image, class_indices):
    img_tensor = preprocess_image_pytorch(image).to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    inference_time = time.time() - start_time
    probs = probabilities.cpu().numpy()[0]
    predicted_idx = np.argmax(probs)
    confidence = probs[predicted_idx] * 100
    predicted_class = class_indices[str(predicted_idx)]
    all_probs = {class_indices[str(i)]: probs[i] * 100 for i in range(len(probs))}
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
    return predicted_class, confidence, all_probs, inference_time

def predict_keras(model, image, class_indices):
    img_array = preprocess_image_keras(image)
    start_time = time.time()
    predictions = model.predict(img_array, verbose=0)
    inference_time = time.time() - start_time
    probs = predictions[0]
    predicted_idx = np.argmax(probs)
    confidence = probs[predicted_idx] * 100
    predicted_class = class_indices[str(predicted_idx)]
    all_probs = {class_indices[str(i)]: probs[i] * 100 for i in range(len(probs))}
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
    return predicted_class, confidence, all_probs, inference_time

# Sidebar
with st.sidebar:
    st.markdown("### MODEL CONFIGURATION")
    st.markdown("<div style='margin-bottom: 0.5rem; color: rgba(255,255,255,0.8); font-size: 0.85rem;'>Select models to use for analysis:</div>", unsafe_allow_html=True)
    selected_models = st.multiselect(
        "Select Models",
        options=list(models_dict.keys()),
        default=list(models_dict.keys()),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### SYSTEM INFO")
    st.markdown(f"""
    <div class="info-container">
        <div class="info-item">
            <span class="info-label">Device</span>
            <span class="info-value">{device}</span>
        </div>
        <div class="info-item">
            <span class="info-label">Classes</span>
            <span class="info-value">{num_classes}</span>
        </div>
        <div class="info-item">
            <span class="info-label">Models</span>
            <span class="info-value">{len(models_dict)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### LOADED MODELS")
    for model_name in models_dict.keys():
        framework = "PyTorch" if "PyTorch" in model_name else "TensorFlow"
        st.markdown(f"""
        <div class="model-list-item">
            <span class="model-check">✓</span>
            <div>
                <div class="model-name">{model_name.split(' (')[0]}</div>
                <div class="model-framework">{framework}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
            Upload a rice image to classify its variety using our AI models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload rice image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.predictions_made = False
            st.session_state.predictions_results = []
        
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
        
        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.caption(f"📐 Dimensions: {image.size[0]} × {image.size[1]} pixels • Format: {image.format}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎯 Analysis Results</div>', unsafe_allow_html=True)
    
    if uploaded_file is None:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #64748b;">
            <p style="font-size: 1.1rem; margin: 0;">Upload an image to see classification results</p>
            <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
    elif not selected_models:
        st.warning("⚠️ Please select at least one model from the sidebar")
    else:
        # Button analyze - hanya muncul jika ada image dan model dipilih
        if st.button("🚀 Analyze Image", type="primary"):
            st.session_state.predictions_results = []
            image = st.session_state.current_image
            
            with st.spinner("Analyzing image..."):
                for model_name in selected_models:
                    model = models_dict[model_name]
                    
                    if 'Keras' in model_name or 'CNN' in model_name:
                        predicted_class, confidence, all_probs, inference_time = predict_keras(model, image, class_indices)
                    else:
                        predicted_class, confidence, all_probs, inference_time = predict_pytorch(model, image, class_indices)
                    
                    st.session_state.predictions_results.append({
                        'model_name': model_name,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'all_probs': all_probs,
                        'inference_time': inference_time
                    })
                
                st.session_state.predictions_made = True
        
        # Tampilkan hasil jika sudah ada prediksi
        if st.session_state.predictions_made and st.session_state.predictions_results:
            for result in st.session_state.predictions_results:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-model">{result['model_name']}</div>
                    <div class="result-class">{result['predicted_class']}</div>
                    <div class="result-confidence">{result['confidence']:.2f}% Confidence</div>
                    <div class="result-time">⚡ {result['inference_time']*1000:.2f}ms</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Top 3 Predictions:**")
                top_3 = list(result['all_probs'].items())[:3]
                
                for i, (class_name, prob) in enumerate(top_3, 1):
                    st.markdown(f'<div class="prob-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="prob-label">{i}. {class_name}</div>', unsafe_allow_html=True)
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.progress(float(prob) / 100)
                    with col_b:
                        st.markdown(f"**{prob:.1f}%**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if result != st.session_state.predictions_results[-1]:
                    st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("👆 Click 'Analyze Image' button above to start classification")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Comparison
if len(selected_models) > 1 and uploaded_file is not None and st.session_state.predictions_made:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 Model Performance Comparison"):
        import pandas as pd
        comparison_data = [{
            'Model': r['model_name'],
            'Prediction': r['predicted_class'],
            'Confidence': f"{r['confidence']:.2f}%",
            'Time': f"{r['inference_time']*1000:.2f}ms"
        } for r in st.session_state.predictions_results]
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

# Training Results Section
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-card">
    <div class="section-header">📊 Model Training Results</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 CNN", "📈 MobileNetV2", "📈 ResNet50"])

with tab1:
    st.markdown("### CNN Training Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training History")
        try:
            st.image("CNN/training_history.png", use_container_width=True)
        except:
            st.info("Training history images not found. Please run the CNN training script first.")
    
    with col2:
        st.markdown("#### Confusion Matrix")
        try:
            st.image("CNN/confusion_matrix.png", use_container_width=True)
        except:
            st.info("Confusion matrix not found. Please run the CNN training script first.")
        
        st.markdown("#### Per-Class Accuracy")
        try:
            st.image("CNN/per_class_accuracy.png", use_container_width=True)
        except:
            st.info("Per-class accuracy chart not found. Please run the CNN training script first.")

with tab2:
    st.markdown("### MobileNetV2 Training Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training History")
        try:
            st.image("Mobile Net V2/training_history.png", use_container_width=True)
        except:
            st.info("Training history images not found. Please run the MobileNetV2 training script first.")
    
    with col2:
        st.markdown("#### Confusion Matrix")
        try:
            st.image("Mobile Net V2/confusion_matrix.png", use_container_width=True)
        except:
            st.info("Confusion matrix not found. Please run the MobileNetV2 training script first.")
        
        st.markdown("#### Per-Class Accuracy")
        try:
            st.image("Mobile Net V2/per_class_accuracy.png", use_container_width=True)
        except:
            st.info("Per-class accuracy chart not found. Please run the MobileNetV2 training script first.")

with tab3:
    st.markdown("### ResNet50 Training Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training History")
        try:
            st.image("RestNet50/training_history.png", use_container_width=True)
        except:
            st.info("Training history images not found. Please run the ResNet50 training script first.")
    
    with col2:
        st.markdown("#### Confusion Matrix")
        try:
            st.image("RestNet50/confusion_matrix.png", use_container_width=True)
        except:
            st.info("Confusion matrix not found. Please run the ResNet50 training script first.")
        
        st.markdown("#### Per-Class Accuracy")
        try:
            st.image("RestNet50/per_class_accuracy.png", use_container_width=True)
        except:
            st.info("Per-class accuracy chart not found. Please run the ResNet50 training script first.")

# About
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️ About This System"):
    st.markdown("""
    ### Multi-Framework Deep Learning Platform
    
    **PyTorch Models:**
    - **MobileNetV2**: Lightweight, fast inference (~91% accuracy)
    - **ResNet50**: Deep architecture, high accuracy (~94% accuracy)
    
    **TensorFlow/Keras:**
    - **Custom CNN**: Purpose-built network (~89% accuracy)
    
    **Features:**
    - Multi-framework support
    - Real-time inference
    - Comparative analysis
    - Transfer learning from ImageNet
    """)

# Footer
st.markdown("""
<div style="text-align: center; color: #9ca3af; padding: 2rem; margin-top: 2rem;">
    <p style="margin: 0; font-weight: 600;">Rice Classification System v2.0</p>
    <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem;">Powered by PyTorch & TensorFlow</p>
</div>
""", unsafe_allow_html=True)