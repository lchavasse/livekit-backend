FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Pre-download models
RUN python -c "from livekit.plugins import silero; silero.VAD.load()"

# Set environment variables
ENV PORT=8080

# Run the application in production mode
CMD ["python", "agent.py", "start"]