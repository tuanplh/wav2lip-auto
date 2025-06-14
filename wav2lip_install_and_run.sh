#!/bin/bash

echo "🔹 Bắt đầu cài đặt Wav2Lip GAN WebUI..."

# 1) Xoá cũ, clone mới
rm -rf Wav2Lip
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip

# 2) Cài gói Python (chống conflict)
pip install torch torchvision numpy==1.23.5 librosa==0.8.1 numba==0.56.4 scipy matplotlib tqdm requests opencv-python-headless==4.9.0.80 gradio==4.29.0

# 3) Tải model GAN + detector
mkdir -p checkpoints
wget -q --show-progress https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth -O checkpoints/wav2lip_gan.pth

mkdir -p face_detection/detection/sfd
wget -q --show-progress https://huggingface.co/rippertnt/wav2lip/resolve/c16701c074d8bab99d1a05379c7771504a9d7fbe/s3fd.pth -O face_detection/detection/sfd/s3fd.pth

# 4) Tạo outputs/
mkdir -p outputs

# 5) Tạo app.py chuẩn (overwrite)
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
    output_name = os.path.join(output_dir, "result_voice.mp4")

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

    if os.path.exists("results/result_voice.mp4"):
        shutil.move("results/result_voice.mp4", output_name)

    return output_name

iface = gr.Interface(
    fn=lip_sync,
    inputs=[
        gr.File(label="Upload Image OR Video"),
        gr.Audio(label="Upload Audio", type="filepath")
    ],
    outputs=gr.Video(label="Lipsynced Video"),
    title="✨ Wav2Lip GAN WebUI — Stable Download",
    description="Upload image/video + audio → Click Run → Download works 100%."
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
EOF

# 6) Chạy Gradio
echo "✅ Mọi thứ đã sẵn sàng. Đang khởi chạy Gradio..."
python3 app.py
