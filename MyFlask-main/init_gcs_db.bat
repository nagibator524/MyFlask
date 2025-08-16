@echo off
REM init_gcs_db.bat

REM This script initializes the database in Google Cloud Storage

echo Initializing database in Google Cloud Storage...

REM Check if GCS_BUCKET_NAME is set
if "%GCS_BUCKET_NAME%"=="" (
    echo Error: GCS_BUCKET_NAME environment variable is not set
    pause
    exit /b 1
)

REM Run the app in initialization mode
python -c "import os; os.environ['GCS_BUCKET_NAME'] = os.environ.get('GCS_BUCKET_NAME', 'your-bucket-name'); from app import init_db; print('Initializing database in GCS bucket:', os.environ.get('GCS_BUCKET_NAME')); init_db(); print('Database initialized successfully!')"

echo Database initialization complete.
pause