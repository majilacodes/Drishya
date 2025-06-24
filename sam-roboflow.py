import streamlit as st
import torch
import numpy as np
import cv2
import os
import requests
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from streamlit_drawable_canvas import st_canvas
from io import BytesIO

# Suppress the torch.classes warning
import warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

@st.cache_resource
def load_model():
    """Load the SAM model by downloading weights automatically"""
    # Check if CUDA is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model configuration
    model_type = "vit_b"
    model_filename = "sam_vit_b_01ec64.pth"
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    # Use a persistent directory for model storage
    home_dir = os.path.expanduser("~")
    model_dir = os.path.join(home_dir, ".cache", "drishya", "models")
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, model_filename)

    # Download the model if it doesn't exist
    if not os.path.exists(checkpoint_path):
        try:
            # Create progress tracking containers
            progress_container = st.container()
            
            with progress_container:
                st.info("🔄 Downloading SAM model weights... This may take 2-3 minutes on first run.")
                progress_bar = st.progress(0)
                status_text = st.empty()

            # Download with improved error handling
            response = requests.get(model_url, stream=True, timeout=120)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
            
            # Use a temporary file with proper cleanup
            temp_path = checkpoint_path + ".tmp"

            try:
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Update progress with better formatting
                            if total_size > 0:
                                progress = min(downloaded / total_size, 1.0)
                                progress_bar.progress(progress)
                                mb_downloaded = downloaded / (1024*1024)
                                mb_total = total_size / (1024*1024)
                                status_text.text(f"Downloaded {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({progress*100:.1f}%)")

                # Verify file size before moving
                if total_size > 0 and downloaded < total_size * 0.95:  # Allow 5% tolerance
                    raise Exception(f"Download incomplete: {downloaded}/{total_size} bytes")
                
                # Move temp file to final location
                os.rename(temp_path, checkpoint_path)
                
                # Clean up progress indicators
                progress_bar.progress(1.0)
                status_text.text("Download complete! Loading model...")
                
            except Exception as e:
                # Clean up temp file on any error
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise e
            finally:
                # Clear progress indicators after a brief delay
                import time
                time.sleep(1)
                progress_container.empty()

        except requests.exceptions.Timeout:
            st.error("❌ Download timeout. Please check your internet connection and try again.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection error. Please check your internet connection and try again.")
            st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Download failed: {str(e)}")
            # Clean up corrupted file if it exists
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                except:
                    pass
            st.stop()

    # Verify file exists and has reasonable size before loading
    if not os.path.exists(checkpoint_path):
        st.error("❌ Model file not found after download.")
        st.stop()
    
    file_size = os.path.getsize(checkpoint_path)
    expected_min_size = 300 * 1024 * 1024  # 300MB minimum
    if file_size < expected_min_size:
        st.error(f"❌ Model file appears corrupted (size: {file_size/(1024*1024):.1f}MB). Please refresh to re-download.")
        try:
            os.remove(checkpoint_path)
        except:
            pass
        st.stop()

    try:
        # Load the model with better error handling
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        mask_predictor = SamPredictor(sam)
        
        # Test model loading
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask_predictor.set_image(test_image)
        
        return mask_predictor, device

    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.error("The model file may be corrupted. Please refresh the page to re-download.")
        # Remove corrupted file
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except:
                pass
        st.stop()

# Set page configuration to wide layout
st.set_page_config(
    page_title="Drishya - AI Product Replacement",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-username/drishya',
        'Report a bug': 'https://github.com/your-username/drishya/issues',
        'About': "Drishya - AI-powered product replacement tool using Meta's SAM"
    }
)

# Password protection
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "setuftw":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Enter the magic words", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # Stop execution if password is incorrect

# Initialize session state for model management with better defaults
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'mask_predictor' not in st.session_state:
    st.session_state.mask_predictor = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'model_load_error' not in st.session_state:
    st.session_state.model_load_error = None

# Load model with improved error handling
if not st.session_state.model_loaded and st.session_state.model_load_error is None:
    try:
        with st.spinner("🔄 Loading AI model... This may take 2-3 minutes on first run."):
            st.session_state.mask_predictor, st.session_state.device = load_model()
            st.session_state.model_loaded = True
        st.success("✅ AI model loaded successfully! You can now upload images.")
    except Exception as e:
        st.session_state.model_load_error = str(e)
        st.error(f"❌ Failed to load model: {str(e)}")
        st.error("Please refresh the page to try again.")
        st.stop()

# Show error if model failed to load previously
if st.session_state.model_load_error is not None:
    st.error(f"❌ Model loading failed: {st.session_state.model_load_error}")
    if st.button("🔄 Retry Loading Model"):
        # Reset error state and try again
        st.session_state.model_load_error = None
        st.session_state.model_loaded = False
        st.rerun()
    st.stop()

# Helper functions
def show_mask(mask, image):
    """Apply mask on image for visualization."""
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    # Create a visualization with mask overlay
    result = image.copy()
    mask_rgb = (mask_image[:,:,:3] * 255).astype(np.uint8)
    mask_alpha = (mask_image[:,:,3:] * 255).astype(np.uint8)
    
    # Blend where mask exists
    alpha_channel = mask_alpha / 255.0
    for c in range(3):
        result[:,:,c] = result[:,:,c] * (1 - alpha_channel[:,:,0]) + mask_rgb[:,:,c] * alpha_channel[:,:,0]
    
    return result

def create_feathered_mask(mask, feather_amount=10):
    """Create a feathered mask with smooth edges for better blending."""
    # Ensure mask is binary
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    binary_mask = (mask > 128).astype(np.uint8)
    
    # Create distance transform from the mask edges
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
    
    # Normalize the distance transform
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Create inverse distance transform for outside the mask
    inv_binary_mask = 1 - binary_mask
    inv_dist_transform = cv2.distanceTransform(inv_binary_mask, cv2.DIST_L2, 3)
    cv2.normalize(inv_dist_transform, inv_dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Create a feathered mask by combining both distance transforms
    feathered_mask = np.ones_like(dist_transform, dtype=np.float32)
    
    # Apply feathering at the boundaries
    feathered_mask = np.where(
        dist_transform > 0,
        np.minimum(1.0, dist_transform * (feather_amount / 2)),
        feathered_mask
    )
    
    feathered_mask = np.where(
        inv_dist_transform < feather_amount,
        np.maximum(0.0, 1.0 - (inv_dist_transform / feather_amount)),
        feathered_mask * binary_mask
    )
    
    return feathered_mask

def apply_color_grading(product_image, target_image, mask, strength=0.5):
    """Apply color grading to make the product match the color tone of the target area."""
    # Ensure mask is binary and has correct shape
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    binary_mask = (mask > 128).astype(np.uint8)
    
    # Get the region of interest from the target image based on the mask
    y_indices, x_indices = np.where(binary_mask == 1)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return product_image  # No adjustment if mask is empty
    
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # Extract the target region
    target_region = target_image[y_min:y_max+1, x_min:x_max+1]
    
    # Create a mask for the target region
    local_mask = binary_mask[y_min:y_max+1, x_min:x_max+1]
    
    # Apply mask to target region to only consider pixels within the mask
    masked_target = target_region.copy()
    for c in range(3):  # Process each color channel
        masked_target[:,:,c] = masked_target[:,:,c] * local_mask
    
    # Calculate mean color of the masked target region
    target_pixels = local_mask.sum()
    if target_pixels == 0:
        return product_image  # No pixels to match
    
    target_means = [
        (masked_target[:,:,c].sum() / target_pixels) for c in range(3)
    ]
    
    # Calculate standard deviation of the target region for each channel
    target_std = [0, 0, 0]
    for c in range(3):
        # Calculate squared differences for non-zero mask pixels
        squared_diffs = np.zeros_like(local_mask, dtype=float)
        squared_diffs[local_mask > 0] = ((masked_target[:,:,c][local_mask > 0] - target_means[c]) ** 2)
        target_std[c] = np.sqrt(squared_diffs.sum() / target_pixels)
    
    # Handle alpha channel if present
    has_alpha = product_image.shape[2] == 4
    if has_alpha:
        alpha_channel = product_image[:,:,3].copy()
        product_rgb = product_image[:,:,:3].copy()
    else:
        product_rgb = product_image.copy()
    
    # Calculate mean and std of the product image (only for non-transparent pixels if has alpha)
    if has_alpha:
        # Only consider pixels that aren't fully transparent
        prod_mask = (alpha_channel > 0).astype(float)
        prod_pixels = prod_mask.sum()
        if prod_pixels == 0:
            return product_image  # No pixels to adjust
        
        product_means = [
            (product_rgb[:,:,c] * prod_mask).sum() / prod_pixels for c in range(3)
        ]
        
        product_std = [0, 0, 0]
        for c in range(3):
            squared_diffs = np.zeros_like(prod_mask, dtype=float)
            valid_pixels = prod_mask > 0
            if valid_pixels.sum() > 0:
                squared_diffs[valid_pixels] = ((product_rgb[:,:,c][valid_pixels] - product_means[c]) ** 2)
                product_std[c] = np.sqrt(squared_diffs.sum() / prod_pixels)
    else:
        h, w = product_rgb.shape[:2]
        prod_pixels = h * w
        product_means = [
            product_rgb[:,:,c].sum() / prod_pixels for c in range(3)
        ]
        
        product_std = [0, 0, 0]
        for c in range(3):
            squared_diffs = (product_rgb[:,:,c] - product_means[c]) ** 2
            product_std[c] = np.sqrt(squared_diffs.sum() / prod_pixels)
    
    # Perform color grading by adjusting mean and standard deviation
    graded_product = product_rgb.copy().astype(float)
    
    for c in range(3):
        # Skip channels with zero std to avoid division by zero
        if product_std[c] == 0:
            continue
            
        # Normalize the product channel
        normalized = (graded_product[:,:,c] - product_means[c]) / product_std[c]
        
        # Apply target statistics with the specified strength
        if strength < 1.0:
            # Blend between original and target values
            adj_std = product_std[c] * (1 - strength) + target_std[c] * strength
            adj_mean = product_means[c] * (1 - strength) + target_means[c] * strength
        else:
            adj_std = target_std[c]
            adj_mean = target_means[c]
            
        # Apply the adjustment
        graded_product[:,:,c] = normalized * adj_std + adj_mean
    
    # Clip values to valid range
    graded_product = np.clip(graded_product, 0, 255).astype(np.uint8)
    
    # Reattach alpha channel if needed
    if has_alpha:
        graded_product_with_alpha = np.zeros((graded_product.shape[0], graded_product.shape[1], 4), dtype=np.uint8)
        graded_product_with_alpha[:,:,:3] = graded_product
        graded_product_with_alpha[:,:,3] = alpha_channel
        return graded_product_with_alpha
    
    return graded_product

def replace_product_in_image(ad_image, new_product, mask, scale_factor=1.0, feather_amount=15, use_blending=True):
    """
    Replace a product in an ad image with improved edge blending and error handling.
    """
    try:
        # Input validation
        if ad_image is None or new_product is None or mask is None:
            raise ValueError("Invalid input: image, product, or mask is None")
        
        if len(ad_image.shape) != 3 or len(new_product.shape) not in [3, 4]:
            raise ValueError("Invalid image dimensions")
        
        # Ensure mask is binary
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        binary_mask = (mask > 128).astype(np.uint8)
        
        # Get bounding box from mask
        y_indices, x_indices = np.where(binary_mask == 1)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return ad_image
            
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        # Validate bounding box
        if y_max <= y_min or x_max <= x_min:
            return ad_image
        
        # Calculate dimensions with bounds checking
        mask_height = min(y_max - y_min + 1, ad_image.shape[0] - y_min)
        mask_width = min(x_max - x_min + 1, ad_image.shape[1] - x_min)
        
        if mask_height <= 0 or mask_width <= 0:
            return ad_image
        
        # Create output image
        output = ad_image.copy()
        
        # Get product dimensions and aspect ratio with validation
        prod_height, prod_width = new_product.shape[:2]
        if prod_height <= 0 or prod_width <= 0:
            return ad_image
            
        prod_aspect_ratio = prod_width / prod_height
        
        # Calculate the dimensions to preserve aspect ratio
        base_dimension = min(mask_width, mask_height)
        scaled_dimension = max(1, base_dimension * scale_factor)
        
        # Calculate new dimensions based on aspect ratio
        if prod_aspect_ratio > 1.0:
            resize_width = scaled_dimension * prod_aspect_ratio
            resize_height = scaled_dimension
        else:
            resize_width = scaled_dimension
            resize_height = scaled_dimension / prod_aspect_ratio
        
        # Ensure minimum dimensions
        resize_width = max(1, int(resize_width))
        resize_height = max(1, int(resize_height))
        
        # Calculate centering offsets
        offset_x = int((mask_width - resize_width) / 2)
        offset_y = int((mask_height - resize_height) / 2)
        
        # Create a feathered mask for better edge blending
        mask_roi = binary_mask[y_min:y_max+1, x_min:x_max+1].astype(np.float32)
        feathered_mask = mask_roi
        if use_blending:
            feathered_mask = create_feathered_mask(mask_roi, feather_amount)
        
        # Handle transparent images (RGBA) vs RGB
        if new_product.shape[2] == 4:
            # RGBA handling with improved error checking
            alpha = new_product[:, :, 3] / 255.0
            rgb = new_product[:, :, :3]
            
            # Resize with proper interpolation
            resized_rgb = cv2.resize(rgb, (resize_width, resize_height), interpolation=cv2.INTER_LANCZOS4)
            resized_alpha = cv2.resize(alpha, (resize_width, resize_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Create product mask with bounds checking
            product_mask = np.zeros((mask_height, mask_width), dtype=np.float32)
            
            # Calculate safe paste coordinates
            paste_y_start = max(0, offset_y)
            paste_y_end = min(mask_height, offset_y + resize_height)
            paste_x_start = max(0, offset_x)
            paste_x_end = min(mask_width, offset_x + resize_width)
            
            # Calculate corresponding product coordinates
            prod_y_start = max(0, -offset_y)
            prod_y_end = min(resize_height, prod_y_start + (paste_y_end - paste_y_start))
            prod_x_start = max(0, -offset_x)
            prod_x_end = min(resize_width, prod_x_start + (paste_x_end - paste_x_start))
            
            # Place the resized alpha with bounds checking
            if (paste_y_end > paste_y_start and paste_x_end > paste_x_start and 
                prod_y_end > prod_y_start and prod_x_end > prod_x_start):
                
                try:
                    product_mask[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = \
                        resized_alpha[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
                except ValueError:
                    # Fallback: center the product
                    center_y = mask_height // 2
                    center_x = mask_width // 2
                    half_h = resize_height // 2
                    half_w = resize_width // 2
                    
                    y_start = max(0, center_y - half_h)
                    y_end = min(mask_height, center_y + half_h)
                    x_start = max(0, center_x - half_w)
                    x_end = min(mask_width, center_x + half_w)
                    
                    product_mask[y_start:y_end, x_start:x_end] = 1.0
            
            # Apply feathered mask
            product_mask = product_mask * feathered_mask
            product_mask_3ch = np.stack([product_mask, product_mask, product_mask], axis=2)
            
            # Get ROI and create RGB blend image
            roi = output[y_min:y_max+1, x_min:x_max+1]
            rgb_to_blend = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
            
            if (paste_y_end > paste_y_start and paste_x_end > paste_x_start and 
                prod_y_end > prod_y_start and prod_x_end > prod_x_start):
                try:
                    rgb_to_blend[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = \
                        resized_rgb[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
                except ValueError:
                    pass  # Skip if dimensions don't match
            
            # Perform blending
            blended_roi = roi * (1 - product_mask_3ch) + rgb_to_blend * product_mask_3ch
            
        else:
            # RGB handling (similar improvements)
            resized_product = cv2.resize(new_product, (resize_width, resize_height), interpolation=cv2.INTER_LANCZOS4)
            
            product_to_blend = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
            
            # Safe coordinate calculation (same as above)
            paste_y_start = max(0, offset_y)
            paste_y_end = min(mask_height, offset_y + resize_height)
            paste_x_start = max(0, offset_x)
            paste_x_end = min(mask_width, offset_x + resize_width)
            
            prod_y_start = max(0, -offset_y)
            prod_y_end = min(resize_height, prod_y_start + (paste_y_end - paste_y_start))
            prod_x_start = max(0, -offset_x)
            prod_x_end = min(resize_width, prod_x_start + (paste_x_end - paste_x_start))
            
            if (paste_y_end > paste_y_start and paste_x_end > paste_x_start and 
                prod_y_end > prod_y_start and prod_x_end > prod_x_start):
                try:
                    product_to_blend[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = \
                        resized_product[prod_y_start:prod_y_end, prod_x_start:prod_x_end]
                except ValueError:
                    pass
            
            # Create product mask
            product_mask = np.zeros((mask_height, mask_width), dtype=np.float32)
            if (paste_y_end > paste_y_start and paste_x_end > paste_x_start):
                product_mask[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = 1
            
            product_mask = product_mask * feathered_mask
            product_mask_3ch = np.stack([product_mask, product_mask, product_mask], axis=2)
            
            roi = output[y_min:y_max+1, x_min:x_max+1]
            blended_roi = roi * (1 - product_mask_3ch) + product_to_blend * product_mask_3ch
        
        # Apply edge blending if enabled
        if use_blending:
            try:
                edge_kernel = np.ones((5, 5), np.uint8)
                edge_mask = cv2.dilate(mask_roi.astype(np.uint8), edge_kernel) - mask_roi.astype(np.uint8)
                edge_mask = np.clip(edge_mask, 0, 1).astype(np.float32)
                
                # Apply guided filtering with fallback to Gaussian blur
                try:
                    r = 5
                    eps = 0.1
                    harmonized_blend = blended_roi.copy()
                    for c in range(3):
                        harmonized_blend[:,:,c] = cv2.guidedFilter(
                            roi[:,:,c].astype(np.float32), 
                            blended_roi[:,:,c].astype(np.float32),
                            r, eps
                        )
                    blended_roi = harmonized_blend.astype(np.uint8)
                except:
                    # Fallback to Gaussian blur
                    blur_amount = 3
                    edge_blur = cv2.GaussianBlur(edge_mask, (blur_amount*2+1, blur_amount*2+1), 0) * 0.7
                    edge_blur_3ch = np.stack([edge_blur, edge_blur, edge_blur], axis=2)
                    harmonized_blend = blended_roi * (1 - edge_blur_3ch) + cv2.GaussianBlur(blended_roi, (blur_amount*2+1, blur_amount*2+1), 0) * edge_blur_3ch
                    blended_roi = harmonized_blend.astype(np.uint8)
            except:
                pass  # Use basic blending if edge processing fails
        
        # Place the blended region back with bounds checking
        try:
            output[y_min:y_max+1, x_min:x_max+1] = blended_roi
        except ValueError:
            # Fallback: try to place what fits
            out_h, out_w = output.shape[:2]
            roi_h, roi_w = blended_roi.shape[:2]
            
            actual_y_end = min(y_min + roi_h, out_h)
            actual_x_end = min(x_min + roi_w, out_w)
            actual_roi_h = actual_y_end - y_min
            actual_roi_w = actual_x_end - x_min
            
            if actual_roi_h > 0 and actual_roi_w > 0:
                output[y_min:actual_y_end, x_min:actual_x_end] = blended_roi[:actual_roi_h, :actual_roi_w]
        
        return output
        
    except Exception as e:
        st.error(f"Error during image processing: {str(e)}")
        return ad_image  # Return original image on error



# Check for health check endpoint
if st.query_params.get("health") == "check":
    st.json({
        "status": "healthy",
        "service": "drishya",
        "model": "SAM",
        "version": "1.0.0",
        "model_loaded": st.session_state.get('model_loaded', False)
    })
    st.stop()

# App title and description
st.title("Drishya - Product Image Replacement Tool")
st.markdown("""
**AI-powered product replacement** using Meta's Segment Anything Model (SAM)

**How it works:**
1. Upload an image containing a product
2. Draw a box around the product to replace
3. Upload a new product image
4. Adjust settings and download the result

*The AI model loads automatically after login for optimal performance.*
""")

# Create session state for storing data between reruns
if 'generated_mask' not in st.session_state:
    st.session_state.generated_mask = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'box_drawn' not in st.session_state:
    st.session_state.box_drawn = None
if 'mask_displayed' not in st.session_state:
    st.session_state.mask_displayed = False
if 'processing_step' not in st.session_state:
    st.session_state.processing_step = 1  # Track the current step

# Step 1: Image upload
st.header("Step 1: Upload Image")
uploaded_ad_file = st.file_uploader("Upload image with product to replace", type=["jpg", "jpeg", "png"], key="ad_image")

if uploaded_ad_file is not None:
    # Use the pre-loaded model from session state
    mask_predictor = st.session_state.mask_predictor
    device = st.session_state.device

    # Read the image
    image = Image.open(uploaded_ad_file)

    # Ensure image is in RGB format for consistent processing
    if image.mode == 'RGBA':
        # Convert RGBA to RGB by compositing over white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    image_np = np.array(image)

    # Save original image to session state
    st.session_state.original_image = image_np.copy()

    # Use the RGB image directly
    image_rgb = image_np.copy()
    
    # Step 2: Draw bounding box
    st.header("Step 2: Draw Bounding Box")
    st.markdown("Draw a box around the product you want to replace")
    
    # Set up a reasonable canvas size based on image dimensions
    h, w = image_rgb.shape[:2]
    canvas_width = min(800, w)
    canvas_height = int(h * (canvas_width / w))
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=2,
        stroke_color="rgba(255, 0, 0, 1)",
        background_image=Image.fromarray(image_rgb),
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key="canvas",
        update_streamlit=True,
    )

    # Check if a bounding box was drawn
    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        
        if objects:
            # Get the box coordinates from the first object
            rect = objects[0]
            box_height = rect.get("height", 0) * (h / canvas_height)
            box_width = rect.get("width", 0) * (w / canvas_width)
            box_left = rect.get("left", 0) * (w / canvas_width)
            box_top = rect.get("top", 0) * (h / canvas_height)
            
            # Convert to the [x_min, y_min, x_max, y_max] format needed by SAM
            x_min = max(0, int(box_left))
            y_min = max(0, int(box_top))
            x_max = min(w, int(box_left + box_width))
            y_max = min(h, int(box_top + box_height))
            
            # Save the box coordinates
            st.session_state.box_drawn = [x_min, y_min, x_max, y_max]
            
            # Display the box coordinates
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Box Coordinates: X({x_min}, {x_max}), Y({y_min}, {y_max})")
            
            with col2:
                if st.button("Generate Mask", key="generate_mask"):
                    try:
                        with st.spinner("Processing image"):
                            # Validate coordinates
                            if x_min >= x_max or y_min >= y_max:
                                st.error("Invalid bounding box. Please draw a proper rectangle.")
                                st.stop()
                            
                            # Set the image for the predictor
                            mask_predictor.set_image(image_rgb)
                            
                            # Generate masks with error handling
                            masks, scores, logits = mask_predictor.predict(
                                box=np.array([x_min, y_min, x_max, y_max]),
                                multimask_output=True
                            )
                            
                            if len(masks) == 0:
                                st.error("Failed to generate mask. Please try a different bounding box.")
                                st.stop()
                            
                            # Get best mask by score
                            best_mask_idx = np.argmax(scores)
                            binary_mask = masks[best_mask_idx].astype(np.uint8) * 255
                            
                            # Validate mask
                            if np.sum(binary_mask) == 0:
                                st.error("Generated mask is empty. Please try a different bounding box.")
                                st.stop()
                            
                            # Save to session state
                            st.session_state.generated_mask = binary_mask
                            st.session_state.mask_displayed = True
                            st.session_state.processing_step = 3
                            
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error generating mask: {str(e)}")
                        st.error("Please try with different settings or images.")

    # Step 3: Show mask and allow product upload
    if st.session_state.mask_displayed and st.session_state.generated_mask is not None:
        st.header("Step 3: Mask Generated")
        
        # Create two columns to display the original image and the mask side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(image_rgb, use_column_width=True)
        
        with col2:
            st.markdown("**Generated Mask**")
            # Create visualization with mask overlay
            mask_vis = show_mask(st.session_state.generated_mask, image_rgb)
            st.image(mask_vis, use_column_width=True)
        
        # Step 4: Product replacement
        st.header("Step 4: Replace Product")
        
        # Upload replacement product image
        uploaded_product = st.file_uploader("Upload New Product Image", type=["png", "jpg", "jpeg"], key="product_image")
        
        if uploaded_product is not None:
            # Read the new product image
            new_product_img = Image.open(uploaded_product)

            # Ensure consistent image format handling
            if new_product_img.mode == 'RGBA':
                # Keep RGBA for products that have transparency
                new_product_np = np.array(new_product_img)
            elif new_product_img.mode != 'RGB':
                # Convert other formats to RGB
                new_product_img = new_product_img.convert('RGB')
                new_product_np = np.array(new_product_img)
            else:
                new_product_np = np.array(new_product_img)
            
            # Create a row for the product preview and settings
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**New Product Image**")
                prod_h, prod_w = new_product_np.shape[:2]
                st.image(new_product_np, width=int(prod_w/2), use_column_width=False)
            
            with col2:
                # Scale factor slider
                scale_factor = st.slider(
                    "Product Scale", 
                    min_value=0.5, 
                    max_value=2.0,
                    value=1.0, 
                    step=0.05,
                    help="Control product size relative to mask (1.0 = exact fit, >1.0 = larger than mask)"
                )
                
                # Blending options
                use_blending = st.checkbox("Use Edge Blending", value=True, help="Enable advanced edge blending (uncheck to keep product edges as-is)")
                
                # Only show feathering slider if blending is enabled
                feather_amount = 15  # Default value
                if use_blending:
                    feather_amount = st.slider(
                        "Edge Feathering Amount", 
                        min_value=0, 
                        max_value=30, 
                        value=15, 
                        step=1,
                        help="Controls the softness of edges (higher = softer transitions)"
                    )
                
                # Collapsible UI for Color Grading Options
                with st.expander("Color Grading Options"):
                    enable_color_grading = st.checkbox("Enable Color Grading", value=True, 
                                                    help="Apply color adjustment to match product color tone with the background")
                    
                    # Only show strength slider if color grading is enabled
                    color_grade_strength = 0.5  # Default value
                    if enable_color_grading:
                        color_grade_strength = st.slider(
                            "Color Grade Strength", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.5, 
                            step=0.05,
                            help="How strongly to apply the color grading (0 = original colors, 1 = full match)"
                        )
                        
                        grading_method = st.radio(
                            "Color Grading Method",
                            ["Match Target Area", "Match Entire Image"],
                            help="Choose whether to match colors with just the target area or the entire image"
                        )
            
            # Replace button
            if st.button("Replace Product"):
                try:
                    with st.spinner("Replacing product..."):
                        # Validate inputs
                        if st.session_state.original_image is None:
                            st.error("Original image not found. Please reload the page.")
                            st.stop()
                        
                        if st.session_state.generated_mask is None:
                            st.error("Mask not found. Please generate a mask first.")
                            st.stop()
                        
                        # Use the original image
                        original_img = st.session_state.original_image.copy()

                        # Ensure new product is in the correct format
                        if len(new_product_np.shape) == 2:  # Grayscale
                            new_product_np = cv2.cvtColor(new_product_np, cv2.COLOR_GRAY2RGB)
                        
                        # Apply color grading if enabled
                        graded_product = new_product_np.copy()
                        if enable_color_grading:
                            try:
                                if grading_method == "Match Target Area":
                                    graded_product = apply_color_grading(
                                        new_product_np, 
                                        original_img, 
                                        st.session_state.generated_mask, 
                                        color_grade_strength
                                    )
                                else:
                                    full_mask = np.ones(original_img.shape[:2], dtype=np.uint8) * 255
                                    graded_product = apply_color_grading(
                                        new_product_np,
                                        original_img,
                                        full_mask,
                                        color_grade_strength
                                    )
                            except Exception as e:
                                st.warning(f"Color grading failed: {str(e)}. Using original product colors.")
                        
                        # Replace the product
                        result_image = replace_product_in_image(
                            original_img,
                            graded_product,
                            st.session_state.generated_mask,
                            scale_factor,
                            feather_amount,
                            use_blending
                        )
                        
                        # Display results
                        st.header("Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Original Image**")
                            display_original = np.clip(original_img, 0, 255).astype(np.uint8)
                            st.image(display_original, use_column_width=True)

                        with col2:
                            st.markdown("**Replaced Product**")
                            display_result = np.clip(result_image, 0, 255).astype(np.uint8)
                            st.image(display_result, use_column_width=True)
                        
                        # Create download with error handling
                        try:
                            display_result = np.clip(result_image, 0, 255).astype(np.uint8)
                            result_pil = Image.fromarray(display_result)

                            img_buffer = BytesIO()
                            result_pil.save(img_buffer, format='PNG', optimize=True)
                            img_buffer.seek(0)

                            st.download_button(
                                label="Download Final Image",
                                data=img_buffer.getvalue(),
                                file_name="product_replaced_image.png",
                                mime="image/png"
                            )
                        except Exception as e:
                            st.error(f"Failed to prepare download: {str(e)}")
                            
                except Exception as e:
                    st.error(f"Failed to replace product: {str(e)}")
                    st.error("Please try with different settings or images.")

# Add footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and Meta's Segment Anything Model (SAM)")