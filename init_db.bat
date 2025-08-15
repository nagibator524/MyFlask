@echo off
REM init_db.bat

REM This script initializes the database in Google Cloud Storage

echo Initializing database in Google Cloud Storage...

REM Run the app in initialization mode
python -c "from app import init_db; print('Initializing database...'); init_db(); print('Database initialized successfully!')"

echo Database initialization complete.
pause