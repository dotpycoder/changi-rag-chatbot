# Hugging Face Spaces — Streamlit app
# HF Spaces expects port 7860 and a non-root user (1000)

FROM python:3.11-slim

# HF Spaces requires this user
RUN useradd -m -u 1000 user
USER user

# PATH so local pip installs are found
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install dependencies first (better layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY --chown=user . .

# HF Spaces uses port 7860
EXPOSE 7860

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Launch Streamlit on port 7860
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]