# Drishya - Product Image Replacement Tool

Drishya is an AI-powered tool that allows you to seamlessly replace products in advertisement images using Meta's Segment Anything Model (SAM).

## Features

- **Automatic Model Download**: SAM model weights are downloaded automatically on first use - no manual setup required
- **Automatic Segmentation**: Draw a bounding box around any product and let SAM generate precise segmentation masks
- **Advanced Blending**: Multiple blending techniques for realistic product replacement
- **Color Grading**: Match the color tone of replacement products with the background
- **Edge Feathering**: Control the softness of edges for seamless integration
- **Interactive UI**: Simple, step-by-step interface built with Streamlit

## How It Works

1. Upload an advertisement image
2. Draw a bounding box around the product you want to replace
3. Generate segmentation mask using SAM
4. Upload a new product image to replace the original
5. Adjust blending settings and download the final image

## Requirements

- Python 3.8+
- PyTorch 2.0.1
- TorchVision 0.15.2
- Segment Anything Model (SAM)
- Streamlit 1.25.0+
- OpenCV
- NumPy
- Matplotlib
- PIL

## Quick Start

### Option 1: Run Locally
```bash
# Clone the repository
git clone <your-repo-url>
cd drishya

# Run the local development script
python run_local.py
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run sam-roboflow.py
```

### Option 3: Verify Deployment Readiness
```bash
python verify_deployment.py
```

**Note**: The SAM model weights (~375MB) will be downloaded automatically on first use. Ensure you have a stable internet connection.

## Deployment

### Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repository
4. Set main file path: `sam-roboflow.py`
5. Click "Deploy"

The app will automatically handle all dependencies and model downloads.

### Local Development

For local development, follow the installation steps above. The app will run on `http://localhost:8501`.

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions and alternative platforms.

## Usage Tips

- For best results, use product images with transparent backgrounds
- Adjust the "Edge Feathering" slider to control the softness of transitions
- Use "Color Grading" to match the replacement product's colors with the target area
- The first model load may take 2-3 minutes due to automatic weight download

## License

This project uses Meta's Segment Anything Model (SAM) which is licensed under the Apache 2.0 license.

## Acknowledgements

- [Meta AI Research](https://ai.meta.com/) for the Segment Anything Model
- [Streamlit](https://streamlit.io/) for the interactive web framework
- [OpenCV](https://opencv.org/) for image processing capabilities

Created with ❤️ using Streamlit and Meta's Segment Anything Model (SAM)