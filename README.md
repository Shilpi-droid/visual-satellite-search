---
title: visual-search-satellite
app_file: gradio_app.py
sdk: gradio
sdk_version: 5.44.1
---
# Satellite Image Similarity Search

**AI-Powered Visual Discovery in Satellite Imagery using BigQuery ML**

A scalable system that finds visually similar satellite images using Google Cloud's multimodal AI embeddings and BigQuery's vector search capabilities. Upload any satellite image and instantly discover similar geographical features, terrain patterns, or land use types across large datasets.

## Overview

This project transforms traditional satellite imagery analysis by enabling semantic search based on visual similarity rather than manual annotation or keyword matching. The system processes large multispectral satellite images, generates AI embeddings, and provides sub-second similarity search through an interactive web interface.

### Key Features

- **Zero-Shot Similarity Search**: Find similar images without training custom models
- **Multispectral Data Support**: Handles 4-band satellite imagery (Blue, Green, Red, NIR)
- **Scalable Architecture**: Built on BigQuery ML for production-scale deployment
- **Interactive Interface**: Real-time search with visual results
- **Sub-Second Performance**: Vector indexing enables fast queries across large datasets

## Architecture

```
Satellite Images (TIFF) → Tiling Pipeline → Cloud Storage → BigQuery ML → Vector Search → Web Interface
```

### Core Components

1. **Preprocessing Pipeline** (`tiling_and_uploading.py`): Converts large satellite images into manageable tiles
2. **BigQuery ML Models**: Generates embeddings using Google's multimodalembedding@001
3. **Vector Index**: High-performance cosine similarity search
4. **Web Interface** (`gradio_app.py`): Interactive search application

## Prerequisites

### Google Cloud Setup

1. **Google Cloud Project** with billing enabled
2. **APIs Enabled**:
   - BigQuery API
   - Cloud Storage API
   - Vertex AI API

3. **Authentication**:
   ```bash
   gcloud auth application-default login
   ```

4. **Required Services**:
   - BigQuery with ML enabled
   - Cloud Storage buckets
   - Vertex AI connection

### Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
google-cloud-bigquery
google-cloud-storage  
google-cloud-aiplatform
rasterio
pillow
numpy
gradio
requests
```

## Quick Start

### 1. Configuration

Update the configuration variables in both scripts:

```python
# In tiling_and_uploading.py and gradio_app.py
PROJECT_ID = "your-project-id"
SOURCE_BUCKET_NAME = "your-source-bucket"
DESTINATION_BUCKET_NAME = "your-processed-bucket"
```

### 2. Data Preparation

Place your large satellite TIFF files in the source bucket:

```bash
gsutil cp *.tiff gs://your-source-bucket/
```

### 3. Run the Pipeline

#### Step 1: Process Images
```bash
python tiling_and_uploading.py
```

This will:
- Tile large satellite images into 256x256 pixel chunks
- Convert multispectral data to RGB
- Upload processed tiles to Cloud Storage

#### Step 2: Setup BigQuery Models

Execute these SQL queries in BigQuery:

**Create Embedding Model:**
```sql
CREATE OR REPLACE MODEL `your-project.satellite_images.embedding_model`
REMOTE WITH CONNECTION `your-project.us.gcs-connection-us`
OPTIONS (
  endpoint = 'multimodalembedding@001'
);
```

**Create Object Table:**
```sql
CREATE OR REPLACE EXTERNAL TABLE `your-project.satellite_images.satellite_image_tiles`
WITH CONNECTION `your-project.us.gcs-connection-us`
OPTIONS (
  object_metadata = 'SIMPLE',
  uris = ['gs://your-processed-bucket/*.png']
);
```

**Generate Embeddings:**
```sql
CREATE OR REPLACE TABLE `your-project.satellite_images.image_embeddings` AS
SELECT
  uri,
  ml_generate_embedding_result AS embedding
FROM
  ML.GENERATE_EMBEDDING(
    MODEL `your-project.satellite_images.embedding_model`,
    TABLE `your-project.satellite_images.satellite_image_tiles`
  );
```

**Create Vector Index:**
```sql
CREATE OR REPLACE VECTOR INDEX image_embeddings_index
ON `your-project.satellite_images.image_embeddings`(embedding)
OPTIONS(
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists": 100}'
);
```

#### Step 3: Launch Interface
```bash
python gradio_app.py
```

Access the web interface at `http://localhost:7860`

## Detailed Usage

### Image Preprocessing

The tiling pipeline handles several challenges:

- **Size Management**: Breaks large images into manageable 256x256 tiles
- **Overlap Strategy**: 64-pixel overlap prevents edge artifacts
- **Band Conversion**: Converts 4-band multispectral to 3-band RGB
- **Normalization**: Adaptive pixel value normalization for consistent appearance

### Search Process

1. **Upload**: User uploads a satellite image through the web interface
2. **Embed**: System generates embedding using the same multimodal model
3. **Search**: Executes cosine similarity search against indexed embeddings
4. **Results**: Displays top similar images with similarity scores

### Performance Tuning

#### Vector Index Optimization
```sql
-- For larger datasets, adjust num_lists
CREATE OR REPLACE VECTOR INDEX image_embeddings_index
ON `your-project.satellite_images.image_embeddings`(embedding)
OPTIONS(
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists": 500}'  -- Increase for more data
);
```

#### Batch Processing
For large datasets, process images in batches to manage memory:

```python
# In tiling_and_uploading.py
MAX_CONCURRENT_UPLOADS = 10  # Adjust based on your quota
```

## File Structure

```
├── tiling_and_uploading.py    # Image preprocessing pipeline
├── gradio_app.py             # Web interface application
├── sql_queries/              
│   ├── 1_create_model.sql    # Embedding model setup
│   ├── 2_create_table.sql    # Object table creation
│   ├── 3_generate_embeddings.sql # Batch embedding generation
│   └── 4_create_index.sql    # Vector index creation
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── config/
    └── config.py            # Configuration settings
```

## Troubleshooting

### Common Issues

**Authentication Errors:**
```bash
# Re-authenticate
gcloud auth application-default login
```

**BigQuery Connection Issues:**
- Ensure your project has BigQuery ML enabled
- Verify the GCS connection exists and has proper permissions

**Memory Issues During Tiling:**
```python
# Reduce tile size if encountering memory issues
TILE_SIZE = 128  # Instead of 256
```

**Slow Search Performance:**
- Check if vector index was created successfully
- Consider increasing `num_lists` for larger datasets
- Verify BigQuery quotas aren't being exceeded

### Performance Monitoring

```sql
-- Check embedding table size
SELECT COUNT(*) as total_embeddings 
FROM `your-project.satellite_images.image_embeddings`;

-- Monitor search performance
SELECT 
  uri,
  distance,
  CURRENT_TIMESTAMP() as search_time
FROM VECTOR_SEARCH(
  TABLE `your-project.satellite_images.image_embeddings`,
  'embedding',
  (SELECT [...] AS embedding),
  top_k => 10
);
```

## Cost Considerations

### BigQuery ML Costs
- Embedding generation: ~$0.001 per image
- Vector search queries: Standard BigQuery compute pricing
- Storage: Standard BigQuery storage rates

### Cloud Storage Costs
- Image storage: Standard Cloud Storage rates
- Data transfer: Egress charges for image downloads

### Optimization Tips
- Use regional buckets to reduce transfer costs
- Enable Cloud Storage lifecycle policies for old tiles
- Monitor BigQuery slot usage during batch processing

## Advanced Configuration

### Custom Embedding Models

To use different embedding models:

```sql
CREATE OR REPLACE MODEL `your-project.satellite_images.custom_embedding_model`
REMOTE WITH CONNECTION `your-project.us.gcs-connection-us`
OPTIONS (
  endpoint = 'textembedding-gecko@003'  -- Alternative model
);
```

### Multi-Resolution Support

Process images at different resolutions:

```python
# In tiling_and_uploading.py
RESOLUTIONS = [256, 512, 1024]  # Multiple tile sizes

for resolution in RESOLUTIONS:
    TILE_SIZE = resolution
    # Process with current resolution
```

### Batch Inference

For processing large datasets:

```sql
-- Process embeddings in batches
CREATE OR REPLACE TABLE `your-project.satellite_images.batch_embeddings` AS
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `your-project.satellite_images.embedding_model`,
  (SELECT * FROM `your-project.satellite_images.satellite_image_tiles` LIMIT 1000)
);
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request with detailed description

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Check the troubleshooting section
- Review Google Cloud documentation
- Open an issue with detailed error information

## Acknowledgments

- Google Cloud AI Platform for multimodal embedding models
- BigQuery ML team for vector search capabilities
- Gradio for the web interface framework