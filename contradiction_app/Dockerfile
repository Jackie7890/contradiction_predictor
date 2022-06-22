FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#RUN python -m pytest contradiction_predictor/contradiction_app/tests/test_train_and_predict.py
#RUN python contradiction_predictor/contradiction_app/train_and_predict.py


CMD ["python", "-m", "pytest", "contradiction_predictor/contradiction_app/tests/test_train_and_predict.py"]
CMD ["python", "contradiction_predictor/contradiction_app/train_and_predict.py"]