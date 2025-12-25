# Jetson Orin Nano / JetPack 6 (L4T R36.4.x)
FROM dustynv/l4t-pytorch:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive

# Make OpenCV GUI work with X11 (cv2.imshow)
# python3-opencv is in Ubuntu "universe", so enable it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
 && add-apt-repository universe \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv \
    libgtk-3-0 \
    libcanberra-gtk3-module \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Force pip to use PyPI (avoid Jetson/NVIDIA mirror issues)
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV QT_X11_NO_MITSHM=1

RUN python3 -m pip install --upgrade pip setuptools wheel

# Install non-OpenCV Python deps first
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Install ultralytics without pulling opencv-python (we use apt python3-opencv)
RUN python3 -m pip install --no-cache-dir --no-deps ultralytics \
 && python3 -m pip install --no-cache-dir \
    pyyaml tqdm matplotlib seaborn pandas scipy pillow py-cpuinfo ultralytics-thop

# App + assets
COPY app.py /app/app.py
COPY models/ /app/models/
COPY config/ /app/config/

# Output mount point
RUN mkdir -p /data/output

CMD ["python3", "/app/app.py"]