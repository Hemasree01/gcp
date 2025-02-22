# Use an official Python runtime as a parent image.
FROM python:3.9-slim

# Set the working directory.
WORKDIR /app

# Copy requirements.txt and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose port 8080 for Cloud Run.
EXPOSE 8080

# Run the Flask application.
CMD ["python", "app.py"]
