# Image Search Application

This is a Flask application for image-based product search and inventory management, designed to run on Google Cloud Run with Google Cloud Storage for data persistence.

## Features

- Image-based product search using deep learning models
- Product inventory management
- Sales tracking
- User authentication

## Deployment to Google Cloud Run

### Prerequisites

1. Install and initialize the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Install [Docker](https://docs.docker.com/get-docker/)
3. Create a Google Cloud project
4. Enable the Cloud Run and Cloud Storage APIs

### Setup

1. Create a Google Cloud Storage bucket:
   ```bash
   gsutil mb gs://your-bucket-name
   ```

2. Set up authentication by creating a service account:
   ```bash
   gcloud iam service-accounts create image-search-sa
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \\
       --member="serviceAccount:image-search-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \\
       --role="roles/storage.objectAdmin"
   gcloud iam service-accounts keys create key.json \\
       --iam-account=image-search-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

### Deploy

1. Build the Docker image:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/image-search
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/image-search \\
       --platform managed \\
       --set-env-vars GCS_BUCKET_NAME=your-bucket-name \\
       --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=key.json
   ```

### Environment Variables

- `GCS_BUCKET_NAME`: The name of your Google Cloud Storage bucket
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account key file (when running locally)

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export GCS_BUCKET_NAME=your-bucket-name
   export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/key.json
   ```

3. Initialize the database in Google Cloud Storage:
   ```bash
   # On Linux/Mac
   ./init_gcs_db.sh
   
   # On Windows
   init_gcs_db.bat
   ```

4. Run the application:
   ```bash
   python app.py
   ```

## Health Check

The application includes a health check endpoint at `/health` for cloud deployment monitoring.

## Notes

- The application uses Google Cloud Storage to store both the SQLite database file and product images
- When running locally, make sure you have valid Google Cloud credentials configured
- The database is automatically initialized on first run
- Fixed issues with database connection handling in Google Cloud environments