import gradio as gr
import base64
import tempfile
import os
from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import storage
from PIL import Image
import io
import requests
import numpy as np
from typing import List, Tuple

# --- CONFIGURATION ---
PROJECT_ID = "a21ai-marketing"
LOCATION = "us-central1"

def get_image_embedding(image_path: str) -> list:
    """Gets a multimodal embedding from an image file."""
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    )
    endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/multimodalembedding@001"
    
    # Read image and encode it to base64
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    encoded_content = base64.b64encode(image_bytes).decode("utf-8")

    # Create the instance using the correct format
    instance = {
        "image": {"bytesBase64Encoded": encoded_content}
    }
    
    response = client.predict(endpoint=endpoint, instances=[instance])
    
    # Extract the embedding from the response
    embedding_vector = [v for v in response.predictions[0]['imageEmbedding']]
    return embedding_vector

def find_similar_images_in_bq(embedding_vector: list) -> List[Tuple[str, float]]:
    """Uses the embedding vector to search for similar images in BigQuery."""
    bq_client = bigquery.Client(project=PROJECT_ID)
    
    # Convert embedding vector to proper SQL array format
    embedding_array = "[" + ",".join(map(str, embedding_vector)) + "]"
    
    sql_query = f"""
    SELECT
      base.uri,
      distance
    FROM
      VECTOR_SEARCH(
        TABLE `a21ai-marketing.satellite_images.image_embeddings`,
        'embedding',
        (SELECT {embedding_array} AS embedding),
        top_k => 10,
        distance_type => 'COSINE'
      )
    """
    
    # Execute the query
    query_job = bq_client.query(sql_query)
    results = query_job.result()
    
    # Return results as list of tuples (uri, distance)
    return [(row.uri, row.distance) for row in results]

def fix_color_tint(image_array: np.ndarray) -> np.ndarray:
    """
    Fix color-tinted images by trying different channel arrangements.
    This handles the case where images were processed with incorrect band ordering.
    """
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        corrected_array = image_array.copy()
        
        # We're getting green tint, let's try a different approach
        # Try converting [G, B, R] -> [R, G, B] (complete reordering)
        corrected_array[:, :, [0, 1, 2]] = corrected_array[:, :, [2, 0, 1]]
        
        return corrected_array
    return image_array

def download_image_from_gcs(gcs_uri: str) -> Image.Image:
    """Download an image from Google Cloud Storage and return as PIL Image."""
    try:
        print(f"Attempting to download: {gcs_uri}")
        
        # Method 1: Try using public URL first (faster and doesn't require auth)
        public_url = gcs_uri.replace('gs://', 'https://storage.googleapis.com/')
        print(f"Trying public URL: {public_url}")
        
        try:
            response = requests.get(public_url, timeout=10)
            if response.status_code == 200:
                print(f"Successfully downloaded via public URL: {len(response.content)} bytes")
                img = Image.open(io.BytesIO(response.content))
                
                # Ensure image is in RGB mode and resize for display
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to a reasonable size for display
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                print(f"Image processed: {img.size}, mode: {img.mode}")
                
                # Convert to numpy array for Gradio
                img_array = np.array(img)
                print(f"Image array shape: {img_array.shape}, dtype: {img_array.dtype}, min: {img_array.min()}, max: {img_array.max()}")
                
                # Fix color tint if present
                # img_array = fix_color_tint(img_array)  # Disabled - showing original images
                
                # Return as numpy array (Gradio prefers this)
                return img_array
            else:
                print(f"Public URL failed with status: {response.status_code}")
        except Exception as e:
            print(f"Public URL method failed: {e}")
        
        # Method 2: Fallback to authenticated GCS download
        if not gcs_uri.startswith('gs://'):
            raise ValueError("Invalid GCS URI")
        
        # Remove gs:// prefix and split bucket and object
        path = gcs_uri[5:]  # Remove 'gs://'
        bucket_name, object_name = path.split('/', 1)
        
        print(f"Trying authenticated download - Bucket: {bucket_name}, Object: {object_name}")
        
        # Initialize GCS client
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        
        # Download image data
        image_data = blob.download_as_bytes()
        print(f"Downloaded via GCS client: {len(image_data)} bytes")
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_data))
        print(f"Image size: {img.size}, mode: {img.mode}")
        
        # Ensure image is in RGB mode and resize for display
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to a reasonable size for display
        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        
        # Convert to numpy array for Gradio
        img_array = np.array(img)
        print(f"GCS Image array shape: {img_array.shape}, dtype: {img_array.dtype}, min: {img_array.min()}, max: {img_array.max()}")
        
        # Fix color tint if present
        # img_array = fix_color_tint(img_array)  # Disabled - showing original images
        
        # Return as numpy array (Gradio prefers this)
        return img_array
    
    except Exception as e:
        print(f"Error downloading image {gcs_uri}: {e}")
        # Return a placeholder image with error info
        return create_placeholder_image(f"Error loading:\n{str(e)[:50]}...")

def create_placeholder_image(text: str) -> Image.Image:
    """Create a placeholder image with text."""
    from PIL import ImageDraw, ImageFont
    
    img = Image.new('RGB', (400, 400), color='lightgray')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw text in the center
    try:
        lines = text.split('\n')
        y_offset = 200 - (len(lines) * 10)
        
        for i, line in enumerate(lines):
            if font:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(line) * 6  # Approximate
                text_height = 11
            
            x = (400 - text_width) // 2
            y = y_offset + (i * 20)
            
            draw.text((x, y), line, fill='black', font=font)
    except Exception as e:
        print(f"Error drawing text on placeholder: {e}")
    
    return img

def search_similar_images(input_image):
    """Main function that handles the image similarity search."""
    if input_image is None:
        return (None, "", None, "", None, "", None, "", None, "", None, "", "❌ Please upload an image first")
    
    try:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            input_image.save(tmp_file.name)
            temp_path = tmp_file.name
        
        print(f"Processing uploaded image: {temp_path}")
        
        # Generate embedding for the input image
        print("Generating embedding...")
        embedding = get_image_embedding(temp_path)
        print(f"Generated embedding with {len(embedding)} dimensions")
        
        # Search for similar images
        print("Searching for similar images...")
        similar_images = find_similar_images_in_bq(embedding)
        print(f"Found {len(similar_images)} similar images")
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Download and prepare result images
        result_images = []
        result_info = []
        
        for i, (uri, distance) in enumerate(similar_images[:6]):  # Show top 6 results
            print(f"Processing result {i+1}: {uri}")
            try:
                img = download_image_from_gcs(uri)
                result_images.append(img)
                
                # Extract filename from URI for display
                filename = uri.split('/')[-1]
                result_info.append(f"{filename}\nDistance: {distance:.4f}")
            except Exception as e:
                print(f"Error processing {uri}: {e}")
                # Add a placeholder for failed downloads
                result_images.append(create_placeholder_image(f"Failed to load\n{uri.split('/')[-1]}"))
                result_info.append(f"Error loading image\nDistance: {distance:.4f}")
        
        # Pad with None if we have fewer than 6 results
        while len(result_images) < 6:
            result_images.append(None)
            result_info.append("")
        
        successful_downloads = len([img for img in result_images if img is not None])
        status_msg = f"✅ Found {len(similar_images)} similar images, displayed {successful_downloads}"
        
        return (
            result_images[0], result_info[0],
            result_images[1], result_info[1],
            result_images[2], result_info[2],
            result_images[3], result_info[3],
            result_images[4], result_info[4],
            result_images[5], result_info[5],
            status_msg
        )
        
    except Exception as e:
        print(f"Error in search_similar_images: {e}")
        error_msg = f"❌ Error: {str(e)}"
        return (None, "", None, "", None, "", None, "", None, "", None, "", error_msg)

# Create the Gradio interface
def create_interface():
    with gr.Blocks(
        title="🛰️ Satellite Image Similarity Search",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .result-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🛰️ Satellite Image Similarity Search
        
        Upload a satellite image to find similar images in our database using AI-powered vector embeddings.
        The system uses Google Cloud's multimodal embedding model to find visually similar satellite imagery.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="📤 Upload Your Satellite Image",
                    height=300
                )
                
                search_btn = gr.Button(
                    "🔍 Search Similar Images",
                    variant="primary",
                    size="lg"
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_label=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Similar Images Found")
                
                # Create a grid of result images
                with gr.Row():
                    with gr.Column():
                        result_img_1 = gr.Image(label="Result 1", height=200)
                        result_info_1 = gr.Textbox(show_label=False, lines=2, interactive=False)
                    
                    with gr.Column():
                        result_img_2 = gr.Image(label="Result 2", height=200)
                        result_info_2 = gr.Textbox(show_label=False, lines=2, interactive=False)
                    
                    with gr.Column():
                        result_img_3 = gr.Image(label="Result 3", height=200)
                        result_info_3 = gr.Textbox(show_label=False, lines=2, interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        result_img_4 = gr.Image(label="Result 4", height=200)
                        result_info_4 = gr.Textbox(show_label=False, lines=2, interactive=False)
                    
                    with gr.Column():
                        result_img_5 = gr.Image(label="Result 5", height=200)
                        result_info_5 = gr.Textbox(show_label=False, lines=2, interactive=False)
                    
                    with gr.Column():
                        result_img_6 = gr.Image(label="Result 6", height=200)
                        result_info_6 = gr.Textbox(show_label=False, lines=2, interactive=False)
        
        # Set up the search button click event
        search_btn.click(
            fn=search_similar_images,
            inputs=[input_image],
            outputs=[
                result_img_1, result_info_1,
                result_img_2, result_info_2,
                result_img_3, result_info_3,
                result_img_4, result_info_4,
                result_img_5, result_info_5,
                result_img_6, result_info_6,
                status_text
            ]
        )
        
        gr.Markdown("""
        ---
        ### 📋 How it works:
        1. **Upload** a satellite image using the upload area
        2. **Click** the "Search Similar Images" button
        3. **View** the top 6 most similar images with their similarity scores
        
        The system uses cosine similarity to find images with similar visual features.
        Lower distance values indicate higher similarity.
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
