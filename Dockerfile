# FROM python:3.12-slim

# WORKDIR /app

# COPY . .

# RUN pip install -r requirements.txt

# EXPOSE 8501

# CMD ["streamlit", "run","app.py"]

FROM python:3.12

# Update and install sqlite3 >= 3.35
RUN apt-get update && \
    apt-get install -y sqlite3 libsqlite3-dev && \
    sqlite3 --version

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Set Streamlit to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
