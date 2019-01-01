# README
## docker-composeの使い方
* イメージを作成して、bashで入る
```
$ docker-compose run --rm api bash
```

* Buildしてcomposerを起動する
```
$ docker-compose build
$ docker-compose up
```

## ディレクトリ構成
```
.
├── Dockerfile
├── README
├── conf
│   └── slack_api_token.env
├── dist
│   ├── torch-0.2.0.post3-cp35-cp35m-macosx_10_7_x86_64.whl
│   └── torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
├── docker-compose.yml
├── plugins
│   ├── __init__.py
│   ├── file
│   │   ├── input_lang.pkl
│   │   └── output_lang.pkl
│   ├── model
│   │   ├── decoder_75000.model
│   │   └── encoder_75000.model
│   ├── my_mention.py
│   └── seq2seq.py
├── requirements
│   ├── osx
│   │   ├── Pipfile
│   │   └── Pipfile.lock
│   └── ubuntu
│       ├── Pipfile
│       └── Pipfile.lock
├── run.py
└── slackbot_settings.py
```

## デプロイ手順
### 初回
```
# kubectlのインストール
$ gcloud components update
$ gcloud components update kubectl

# https://console.cloud.google.com/kubernetes/list?project=kenchin-develop
# 上記サイトへアクセスし、Kubernetes Engineの使用を許可する

# クラスタの作成
$ gcloud container clusters create --num-nodes=2 slack-bot-cluster \
--zone asia-northeast1-a \
--enable-autoscaling --min-nodes=2 --max-nodes=5

# imageのbuild
$ docker-compose build
```

## 参考資料
* https://qiita.com/yusukixs/items/11601607c629295d31a7