# syntax=docker/dockerfile:1

FROM python:3.6

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY app.py .

COPY utils.py .

COPY Model/bert_1_epoch.pth .

CMD [ "python3", "app.py"]