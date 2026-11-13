# CPU image (works on Apple Silicon and x86_64).
# Model weights are NOT baked in: they are downloaded on first start into
# /hf-cache, which should be mounted as a volume.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/hf-cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_FLAGGING_MODE=never \
    TORCH_DTYPE=bfloat16

WORKDIR /app

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /hf-cache \
    && chown app:app /hf-cache
COPY --chown=app:app app.py .

USER app
EXPOSE 7860
VOLUME ["/hf-cache"]

# Model loading takes a while (and the first start also downloads ~7.5 GB).
HEALTHCHECK --interval=30s --timeout=10s --start-period=2h --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=5)"

CMD ["python", "app.py"]
