FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

#  No interaction in install
ARG DEBIAN_FRONTEND=noninteractive

# Install base utils
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential net-tools iputils-ping freeglut3-dev libgtk2.0-dev \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirment
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "streamlit_app.py"]
