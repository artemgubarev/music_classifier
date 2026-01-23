FROM python3.11-slim

RUN apt-get update && apt-get install -y 
    ffmpeg 
    libsndfile1 
    && rm -rf varlibaptlists

WORKDIR app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [python, bot.py]