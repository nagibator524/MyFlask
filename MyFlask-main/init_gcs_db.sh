#!/bin/bash
# init_gcs_db.sh

# This script initializes the database in Google Cloud Storage

echo "Initializing database in Google Cloud Storage..."

# Check if GCS_BUCKET_NAME is set
if [ -z "$GCS_BUCKET_NAME" ]; then
    echo "Error: GCS_BUCKET_NAME environment variable is not set"
    exit 1
fi

# Run the app in initialization mode
python -c "
import os
os.environ['GCS_BUCKET_NAME'] = os.environ.get('GCS_BUCKET_NAME', 'your-bucket-name')
from app import init_db
print('Initializing database in GCS bucket:', os.environ.get('GCS_BUCKET_NAME'))
init_db()
print('Database initialized successfully!')
"

echo "Database initialization complete."