# test_gcs.py
import os
from google.cloud import storage

def test_gcs_connection():
    """Test Google Cloud Storage connection"""
    try:
        # Get bucket name from environment variable
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        if not bucket_name:
            print("GCS_BUCKET_NAME environment variable not set")
            return False
            
        # Initialize client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Test if bucket exists
        if bucket.exists():
            print(f"Successfully connected to bucket: {bucket_name}")
            
            # List some blobs to verify access
            blobs = list(bucket.list_blobs(max_results=5))
            print(f"Found {len(blobs)} blobs in bucket:")
            for blob in blobs:
                print(f"  - {blob.name}")
            
            return True
        else:
            print(f"Bucket {bucket_name} does not exist")
            return False
    except Exception as e:
        print(f"Error connecting to Google Cloud Storage: {e}")
        return False

if __name__ == "__main__":
    print("Testing Google Cloud Storage connection...")
    success = test_gcs_connection()
    if success:
        print("GCS connection test passed!")
    else:
        print("GCS connection test failed!")