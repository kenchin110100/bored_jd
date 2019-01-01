FROM python:3.5-stretch
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9

RUN apt-get install -y mecab libmecab-dev mecab-ipadic-utf8

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install pipenv

ENV PIPENV_VENV_IN_PROJECT=true

COPY run.py slackbot_settings.py ./home/
COPY requirements/ubuntu/Pipfile requirements/ubuntu/Pipfile.lock ./home/
COPY dist/ ./home/dist/
COPY plugins/ ./home/plugins/

WORKDIR /home
RUN git clone https://github.com/neologd/mecab-ipadic-neologd.git
WORKDIR /home/mecab-ipadic-neologd
RUN bin/install-mecab-ipadic-neologd -n -y -p /var/lib/mecab/dic/mecab-ipadic-neologd

WORKDIR /home
RUN pipenv install --system

CMD ["python", "run.py"]