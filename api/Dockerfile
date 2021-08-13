FROM python:3.7.0

WORKDIR /app

RUN pip install -U pip wheel cmake
RUN pip install numpy flask xgboost

COPY . .

EXPOSE 5000

CMD [ "python", "run.py"]
