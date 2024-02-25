# .ipynb実行する際のステージ
FROM jupyter/datascience-notebook as jupyter
WORKDIR /app
COPY ./app/solution/submit/requirements.txt .
# xfeatを使うために使用
RUN pip install -U "setuptools<58" 
RUN pip install -r requirements.txt

# # .py実行する際のステージ
# FROM python:3.10 as python
# WORKDIR /app
# COPY ./app/solution/submit/requirements.txt .
# # xfeatを使うために使用(使わないならなくてもOK)
# RUN pip install -U "setuptools<58" 
# RUN pip install -r requirements.txt
