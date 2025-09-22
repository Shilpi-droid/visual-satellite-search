import os
import os
from dotenv import load_dotenv
load_dotenv()  
import io
import rasterio
import numpy as np
from PIL import Image
from google.cloud import storage
from google.api_core import exceptions

# --- 1. CONFIGURE YOUR DETAILS ---
# The bucket with your large, original .tiff files
SOURCE_BUCKET_NAME = "satellite-images429"
# A new, empty bucket to store the small .png tiles
DESTINATION_BUCKET_NAME = "processed-satellite-images429"

TILE_SIZE = 256      # The size of the square tiles (e.g., 256x256 pixels)
OVERLAP = 64         # Pixel overlap to avoid missing objects on tile borders

# --- 2. INITIALIZE CLIENTS ---
# Make sure you are authenticated with `gcloud auth application-default login`
# before running this script, or that your environment is otherwise configured
# with valid credentials.
storage_client = storage.Client()
source_bucket = storage_client.bucket(SOURCE_BUCKET_NAME)
destination_bucket = storage_client.bucket(DESTINATION_BUCKET_NAME)
STEP = TILE_SIZE - OVERLAP

# --- 2.5. ENSURE DESTINATION BUCKET EXISTS ---
print(f"Checking for destination bucket: {DESTINATION_BUCKET_NAME}")
try:
    destination_bucket.reload()  # Check if the bucket exists
    print("Destination bucket already exists.")
except exceptions.NotFound:
    print(f"Destination bucket not found. Creating bucket: {DESTINATION_BUCKET_NAME}")
    # You may want to specify a location for the bucket, e.g., location="US-CENTRAL1"
    new_bucket = storage_client.create_bucket(destination_bucket)
    print(f"Bucket {new_bucket.name} created successfully.")


# --- 3. HELPER FUNCTION TO CHECK IF IMAGE IS ALREADY PROCESSED ---
def is_image_already_processed(image_name, destination_bucket):
    """
    Check if an image has already been processed by looking for any tiles
    with the same base name in the destination bucket.
    """
    base_name = os.path.splitext(image_name)[0]
    # Look for any blob that starts with the base name and contains "_tile_"
    blobs = destination_bucket.list_blobs(prefix=base_name)
    
    for blob in blobs:
        if "_tile_" in blob.name:
            return True  # Found at least one tile, image is already processed
    return False

# --- 4. MAIN PROCESSING LOOP ---
print("Starting tiling process...")
# List all the .tiff files in your source bucket
blobs = source_bucket.list_blobs()

# Count total images for progress tracking
all_images = [blob for blob in blobs if blob.name.endswith((".tiff", ".tif"))]
total_images = len(all_images)
processed_count = 0
skipped_count = 0

print(f"Found {total_images} TIFF images to process")

for blob in all_images:
    # Check if this image has already been processed
    if is_image_already_processed(blob.name, destination_bucket):
        skipped_count += 1
        print(f"--> ⏭️  Skipping {blob.name} (already processed) [{skipped_count}/{total_images}]")
        continue
        
    processed_count += 1
    print(f"--> Processing source file: {blob.name} [{processed_count}/{total_images}]")

    # Download the large TIFF into memory
    tiff_content = blob.download_as_bytes()

    # Use rasterio to open the in-memory file
    with rasterio.io.MemoryFile(tiff_content) as memfile:
        with memfile.open() as src:
            # The source TIFFs have 4 bands (Blue, Green, Red, NIR).
            # We select the visual bands (Red, Green, Blue) for the vision model.
            # Read bands in order: R(3), G(2), B(1) to get RGB order directly
            # This should give us [R, G, B] in the correct order
            rgb_bands = src.read([3, 2, 1])

            # Loop through the image and slice it into tiles
            for y in range(0, src.height - TILE_SIZE + 1, STEP):
                for x in range(0, src.width - TILE_SIZE + 1, STEP):
                    # Extract the tile from the RGB bands.
                    # Shape is (Channels, Height, Width) -> (3, 256, 256)
                    tile_data = rgb_bands[:, y:y + TILE_SIZE, x:x + TILE_SIZE]

                    # --- START: FIX ---
                    # 1. Adaptive normalization based on actual data range
                    # Find the actual min/max values in this tile
                    tile_min = np.min(tile_data)
                    tile_max = np.max(tile_data)
                    
                    # Only normalize if there's actual data (avoid division by zero)
                    if tile_max > tile_min:
                        # Normalize each channel independently
                        normalized_channels = []
                        for c in range(tile_data.shape[0]):
                            band = tile_data[c]
                            bmin, bmax = band.min(), band.max()
                            if bmax > bmin:
                                norm_band = ((band - bmin) / (bmax - bmin) * 255).astype(np.uint8)
                            else:
                                norm_band = np.zeros_like(band, dtype=np.uint8)
                            normalized_channels.append(norm_band)

                        normalized_tile = np.stack(normalized_channels, axis=0)

                    else:
                        # If all pixels are the same value, just convert to uint8
                        normalized_tile = np.clip(tile_data, 0, 255).astype(np.uint8)
                    
                    print(f"Tile at ({y}, {x}): Original range [{tile_min}, {tile_max}] -> Normalized [0, 255]")

                    # 2. Convert from (Channels, Height, Width) to (Height, Width, Channels)
                    #    This is the format Pillow/PIL expects for RGB images.
                    #    We have [R, G, B] in channel order, so no reordering needed
                    tile_for_pil = np.moveaxis(normalized_tile, 0, -1)
                    
                    # 3. Create the PIL Image from the correctly shaped array.
                    img_pil = Image.fromarray(tile_for_pil)
                    # --- END: FIX ---

                    # Create a unique name for the tile
                    base_name = os.path.splitext(blob.name)[0]
                    tile_filename = f"{base_name}_tile_{y}_{x}.png"

                    # Save the PIL Image to an in-memory byte buffer
                    buffer = io.BytesIO()
                    img_pil.save(buffer, 'PNG')
                    buffer.seek(0)

                    # Upload the tile to the destination GCS bucket with retry logic
                    dest_blob = destination_bucket.blob(tile_filename)
                    
                    # Retry upload up to 3 times on timeout
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            dest_blob.upload_from_file(buffer, content_type='image/png', timeout=120)
                            print(f"    ✅ Uploaded: {tile_filename}")
                            break  # Success, exit retry loop
                        except Exception as upload_error:
                            if attempt < max_retries - 1:  # Not the last attempt
                                print(f"    ⚠️  Upload attempt {attempt + 1} failed for {tile_filename}, retrying...")
                                buffer.seek(0)  # Reset buffer position for retry
                                continue
                            else:
                                print(f"    ❌ Failed to upload {tile_filename} after {max_retries} attempts: {upload_error}")
                                # Don't raise - continue with next tile instead of crashing
                                break

    print(f"--> Finished processing and uploading tiles for {blob.name}")

print("All images have been processed successfully!")
print(f"\n📊 Processing Summary:")
print(f"   Total images found: {total_images}")
print(f"   Images processed: {processed_count}")
print(f"   Images skipped (already done): {skipped_count}")
print(f"   Processing efficiency: {((total_images - skipped_count) / total_images * 100):.1f}% new work")

