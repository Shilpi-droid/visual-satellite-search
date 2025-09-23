-- Creates a high-speed index on the embedding column
CREATE OR REPLACE VECTOR INDEX image_embeddings_index
ON `a21ai-marketing.satellite_images.image_embeddings`(embedding)
OPTIONS(
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists": 100}'
);