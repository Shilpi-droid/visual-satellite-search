-- Re-creates the object table with the correct GCS bucket name
CREATE OR REPLACE EXTERNAL TABLE `a21ai-marketing.satellite_images.satellite_image_tiles`
WITH CONNECTION `a21ai-marketing.us.gcs-connection-us`
OPTIONS (
  object_metadata = 'SIMPLE',
  uris = ['gs://processed-satellite-images429/*.png'] -- Corrected bucket name
);