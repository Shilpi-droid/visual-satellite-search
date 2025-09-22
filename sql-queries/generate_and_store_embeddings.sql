-- This will now read from the correct source and generate the embeddings
CREATE OR REPLACE TABLE `a21ai-marketing.satellite_images.image_embeddings` AS
SELECT
  uri,
  ml_generate_embedding_result AS embedding
FROM
  ML.GENERATE_EMBEDDING(
    MODEL `a21ai-marketing.satellite_images.embedding_model`,
    TABLE `a21ai-marketing.satellite_images.satellite_image_tiles`
  );