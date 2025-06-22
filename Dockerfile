# Dockerfile

# Stage 1: Build dependencies and app
# Use a specific, recent patch version of Python 3.11 slim-bookworm.
# 'bookworm' (Debian 12) is modern and generally has newer system libraries (like sqlite3).
# Pinning to a specific patch (e.g., 3.11.9) reduces variability and often includes security fixes.
FROM python:3.11-slim-bullseye



# Set the working directory inside the container
WORKDIR /app

# Copy your requirements.txt file into the container
# This allows Docker to cache this layer, so it only rebuilds dependencies if requirements.txt changes.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir reduces the size of the image by not storing pip's cache.
# --upgrade ensures packages are updated if existing.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire local project directory into the container's /app directory.
# This includes your main.py, database/, sidebar_chat_list.py, alex_characteristics/, files/, .streamlit/ etc.
COPY . .
ENV PYTHONUNBUFFERED 1
# Expose the port that Streamlit will run on (default is 8501)
EXPOSE 8501

# Define the command to run your Streamlit application when the container starts.
# --server.port=8501 is the default and good practice.
# --server.enableCORS=true is crucial for embedding in iframes (like Qualtrics).
# --server.enableXsrfProtection=false is sometimes necessary when embedding in specific external domains.
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=true", "--server.enableXsrfProtection=false"]

# Important Security Note:
# - Ensure .env and .streamlit/secrets.toml are NOT committed to your public GitHub repo.
# - For deployment, pass sensitive keys as environment variables to your Docker container
#   or use your deployment platform's secure secret management features.