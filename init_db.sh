#!/bin/bash
# init_db.sh

# This script initializes the database in Google Cloud Storage

echo "Initializing database in Google Cloud Storage..."

# Run the app in initialization mode
python -c "
from app import init_db
print('Initializing database...')
init_db()
print('Database initialized successfully!')
"