# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install git (needed for some packages)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Expose the port that the application will run on
EXPOSE 8080

# Set environment variables
ENV PORT=8080

# Run the application
CMD ["python", "app.py"]