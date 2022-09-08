FROM python:3.8.0

RUN apt-get update \
    && apt-get upgrade -y \
    && pip install --upgrade pip

WORKDIR /home/akihiro/Project/Natural-Language-Processing-with-Disaster-Tweets/src/


COPY requirement.txt ${PWD}

RUN pip install -r requirement.txt


WORKDIR /home
