#!/bin/bash

echo "==============================="
echo "🚀 Wav2Lip GAN WebUI FINAL REMUX — AUTO RUN 🚀"
echo "==============================="

# 1) Cài ffmpeg + dos2unix
echo "📦 Installing ffmpeg and dos2unix..."
apt update && apt install -y ffmpeg dos2unix

# 2) Clone Wav2Lip
echo "📦 Cloning Wav2Lip repo..."
rm -rf Wav2Lip
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip

# 3) Cài Python packages
echo "📦 Installing Python packages..."
pip install torch torchvision numpy==1.23.5 librosa==0.8.1 numba==0.56.4 scipy matplotlib tqdm requests opencv-python-headless==4.9.0.80 gradio==3.50.2

# 4) Tải model GAN + face detector
echo "📥 Downloading GAN model & face detector..."
mkdir -p checkpoints
wget -q --show-progress https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth -O checkpoints/wav2lip_gan.pth

mkdir -p face_detection/detection/sfd
wget -q --show-progress https://huggingface.co/rippertnt/wav2lip/resolve/c16701c074d8bab99d1a05379c7771504a9d7fbe/s3fd.pth -O face_detection/detection/sfd/s3fd.pth

# 5) Tạo outputs/
echo "📁 Creating outputs/ folder..."
mkdir -p outputs

# 6) Ghi app.py chuẩn REMUX
echo "📝 Writing app.py with REMUX..."
cat > app.py << 'EOF'
import os
import shutil
import subprocess
import gradio as gr

def lip_sync(face, audio):
    face_input_path = face.name
    audio_input_path = audio  # because type="filepath"

    face_ext = os.path.splitext(face_input_path)[1].lower()
    face_name = "input_face" + face_ext
    audio_name = "input_audio.wav"

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    raw_output = "results/result_voice.mp4"
    final_output = os.path.join(output_dir, "result_voice_final.mp4")

    shutil.copy(face_input_path, face_name)
    shutil.copy(audio_input_path, audio_name)

    cmd = [
        "python3", "inference.py",
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
        "--face", face_name,
        "--audio", audio_name,
        "--resize_factor", "1"
    ]

    if face_ext in ['.jpg', '.jpeg', '.png']:
        cmd += ["--static", "True"]

    subprocess.run(cmd, check=True)

    if os.path.exists(raw_output):
        # Remux to ensure proper container
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_output,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            final_output
        ], check=True)
        return final_output
    else:
        return "ERROR: Output video not found"

iface = gr.Interface(
    fn=lip_sync,
    inputs=[
        gr.File(label="Upload Image or Video"),
        gr.Audio(label="Upload Audio", type="filepath")
    ],
    outputs=gr.Video(label="Lipsynced Video"),
    title="✨ Wav2Lip GAN WebUI — FINAL REMUX",
    description="Upload image/video + audio → Click Run → Download always works."
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
EOF

# 7) Đảm bảo line ending LF
dos2unix app.py

# 8) Chạy Gradio
echo "✅ All ready! Starting Gradio WebUI FINAL REMUX..."
python3 app.py
