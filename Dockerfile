FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt /app/

# Install system deps (like awscli) – optional, remove if you don't actually need it
RUN apt-get update \
    && apt-get install -y --no-install-recommends awscli \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app
COPY . /app

# If your main file is application.py:
CMD ["python", "app.py"]

# If instead it's app.py, use this:
# CMD ["python", "app.py"]
