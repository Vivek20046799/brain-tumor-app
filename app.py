import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

# ================= MODEL =================
beta = 0.5
spike_grad = surrogate.fast_sigmoid()

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch//reduction),
            nn.ReLU(),
            nn.Linear(ch//reduction, ch),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch)

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return self.se(torch.relu(out))

class BigSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResBlock(3, 64)
        self.layer2 = ResBlock(64, 128)
        self.layer3 = ResBlock(128, 256)
        self.layer4 = ResBlock(256, 256)

        self.pool = nn.MaxPool2d(2)

        self.lif_mid1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif_mid2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x, T=4):
        mem_mid1 = self.lif_mid1.init_leaky()
        mem_mid2 = self.lif_mid2.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        outputs = []
        x_seq = x.unsqueeze(0).repeat(T,1,1,1,1)

        for step in range(T):
            x_t = x_seq[step]

            x1 = self.pool(self.layer1(x_t))
            spk_mid1, mem_mid1 = self.lif_mid1(x1, mem_mid1)

            x2 = self.pool(self.layer2(spk_mid1))
            spk_mid2, mem_mid2 = self.lif_mid2(x2, mem_mid2)

            x3 = self.pool(self.layer3(spk_mid2))
            x4 = self.pool(self.layer4(x3))

            spk1, mem1 = self.lif1(x4, mem1)

            gap = torch.mean(spk1, dim=[2,3])   

            gap = self.dropout(gap)

            fc = self.fc1(gap)
            
            spk2, mem2 = self.lif2(fc, mem2)

            out = self.fc2(spk2)
            outputs.append(out)

        return torch.stack(outputs).mean(0)

# ================= STREAMLIT CONFIG & UI =================
st.set_page_config(page_title="BigSNN | Vivek Katuri", layout="wide", page_icon="🧠")

# Custom CSS for Dark Theme and Centered Elements
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    .centered-header { text-align: center; color: #00d2ff; margin-bottom: 0px; }
    .subtitle { text-align: center; color: #8b949e; font-size: 1.1rem; margin-top: -10px; margin-bottom: 30px; }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a1c24 0%, #0e1117 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #00d2ff;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.2);
        margin-bottom: 20px;
    }
    .sidebar-branding { text-align: center; padding: 10px 0; border-bottom: 1px solid #30363d; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar: Logo & Personal Details
with st.sidebar:
    # Branding Section
    st.markdown("<div class='sidebar-branding'>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/144/artificial-intelligence.png", width=120)
    st.markdown("<h3 style='color: #00d2ff; margin-top:10px;'>IIT PATNA</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Personal Info
    st.subheader("Developer Profile")
    st.info(f"""
    **Name:** Vivek Katuri  
    **Roll No:** 2301ai36  
    **Branch:** AI & Data Science
    """)
    st.divider()
    st.caption("Model: BigSNN (Leaky-LIF)")
    st.caption("Inference: CPU/Real-time")

# Main Page Branding
st.markdown("<h1 class='centered-header'>🧠 BigSNN: Neuro-Diagnostic Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Spiking Neural Network for Brain Tumor Classification</p>", unsafe_allow_html=True)

# ================= MODEL UTILS =================
@st.cache_resource
def load_model():
    model = BigSNN()
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
        model.eval()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
    return model

model = load_model()
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ================= DASHBOARD LAYOUT =================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader(" Scan Input")
    uploaded_file = st.file_uploader("Upload MRI Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Subject MRI Scan", use_container_width=True)

with col2:
    st.subheader(" Inference Results")
    if uploaded_file:
        img = transform(image).unsqueeze(0)
        
        with st.status("Computing Spike Gradients...", expanded=False) as status:
            with torch.no_grad():
                output = model(img)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
            status.update(label="Analysis Complete", state="complete", expanded=False)

        result = classes[pred.item()]
        confidence = conf.item() * 100

        # Custom Result Display
        st.markdown(f"""
            <div class="prediction-card">
                <h2 style="color: #00d2ff; margin-bottom: 0;">{result}</h2>
                <p style="color: #ffffff; opacity: 0.8; font-size: 1.1rem;">Inference Confidence: {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

        # Clinical Guidance
        if result == "No Tumor":
            st.success(" Analysis shows normal brain morphology.")
        else:
            st.warning(f" Detected features characteristic of **{result}**. Prompt neurosurgical review is advised.")

        # Analytics Chart
        st.write("---")
        st.write("**Spiking Probability Distribution:**")
        chart_data = pd.DataFrame(probs.numpy()[0], index=classes, columns=["Spike Threshold"])
        st.bar_chart(chart_data, color="#00d2ff")
    else:
        st.info("System Ready. Please upload a scan to initiate neural inference.")

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #4b5563; font-size: 0.8rem;'>"
    f"© 2026 Vivek Katuri | Roll No: 2301ai36 | Indian Institute of Technology Patna"
    f"</div>", 
    unsafe_allow_html=True
)
