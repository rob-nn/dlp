FROM tensorflow/tensorflow:2.2.0-gpu

EXPOSE 8888

RUN pip install jupyter

WORKDIR	/projects/dlp

CMD jupyter-notebook --allow-root --ip 0.0.0.0
