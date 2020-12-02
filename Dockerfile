FROM tensorflow/tensorflow:2.2.0-gpu

COPY download_datasets.py /app/
RUN python /app/download_datasets.py

CMD [ "/bin/bash" ]
