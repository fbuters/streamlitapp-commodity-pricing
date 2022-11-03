#Specifying the base image
FROM python:3.9

WORKDIR /app

# Copy only relevant files, I know I should clean up the folder structure...
COPY app.py app.py
COPY requirements.txt requirements.txt 
COPY commodity-prices-2016.csv commodity-prices-2016.csv

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]