# Deployment Guide for Drishya

This guide explains how to deploy Drishya on various platforms.

## Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Streamlit deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set the main file path: `sam-roboflow.py`
   - Click "Deploy"

3. **Configuration**
   - The app will automatically install dependencies from `requirements.txt`
   - System packages from `packages.txt` will be installed
   - Model weights will be downloaded automatically on first use

### Important Notes

- **Memory Requirements**: The SAM model requires significant memory (~2GB). Streamlit Cloud should handle this, but initial loading may take 2-3 minutes.
- **Download Time**: The model weights (~375MB) are downloaded on first use, which may take 1-2 minutes depending on connection speed.
- **Caching**: The model is cached using `@st.cache_resource`, so subsequent uses will be faster.

## Alternative Deployment Options

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "sam-roboflow.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Heroku Deployment

1. Create `Procfile`:
   ```
   web: streamlit run sam-roboflow.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Create `runtime.txt`:
   ```
   python-3.11.0
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Performance Optimization

### For Production Deployment

1. **Model Caching**: The app uses `@st.cache_resource` to cache the loaded model
2. **Memory Management**: Consider using a smaller model variant if memory is limited
3. **CDN**: For faster model downloads, consider hosting the model weights on a CDN

### Monitoring

- Monitor memory usage during model loading
- Set up health checks for the `/health` endpoint (if implemented)
- Monitor download times for model weights

## Troubleshooting

### Common Issues

1. **Memory Errors**: Increase available memory or use a smaller model
2. **Download Failures**: Check internet connectivity and firewall settings
3. **OpenCV Issues**: Ensure all system packages from `packages.txt` are installed

### Logs

Check Streamlit logs for detailed error messages:
```bash
streamlit run sam-roboflow.py --logger.level=debug
```
