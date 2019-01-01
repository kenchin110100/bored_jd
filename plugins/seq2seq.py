# coding: utf-8
"""
chat_botの中身
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from filer3 import Filer

import MeCab

use_cuda = torch.cuda.is_available()

MAX_LENGTH = 20
NEOLOGD_PATH = "/var/lib/mecab/dic/mecab-ipadic-neologd"

def parsing(sentence):
    """
    形態素解析して単語をparse
    """
    mecab = MeCab.Tagger("-d {}".format(NEOLOGD_PATH))
    mecab.parse('')
    res = mecab.parseToNode(sentence)
    list_words = []
    while res:
        list_words.append(res.surface)
        res = res.next
    return [w for w in list_words if w != '']


def indexesFromSentence(lang, words):
    return [lang.word2index[word] if word in lang.word2index else lang.word2index['<unk>'] for word in words]


def variableFromSentence(lang, words):
    indexes = indexesFromSentence(lang, words)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    """
    入力文に対して文を出力する関数
    """
    # 形態素解析して、文を単語のリストに変換
    words = parsing(sentence)
    # 単語をidに変換し、pytorchの関数に変換
    input_variable = variableFromSentence(input_lang, words)
    # 語彙数
    input_length = input_variable.size()[0]
    # encoderの隠れ層を初期化
    encoder_hidden = encoder.initHidden()
    # outputベクトルを初期化
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    # encoderに入力文を読み込ませる
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    # decoderのinputを初期化
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    # decoderの隠れ層をencoderの隠れ層と同じものにする
    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    # デコーダーから文を出力
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            # デコーダーから出力されたidを辞書で単語に変換
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def reply_message(sentence, encoder, decoder, input_lang, output_lang, n=20):
    """
    入力文に対して単語を出力する
    """
    output_words, attentions = evaluate(encoder, decoder, sentence, input_lang, output_lang)
    output_sentence = ' '.join(output_words[:-1])
    return output_sentence


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        # 先頭文字をSOS、文末文字をEOSとする
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, words):
        for word in words:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class EncoderRNN(nn.Module):
    """
    Encoderのクラス
    input_size: 単語数
    hidden_size: 隠れ層のサイズ
    n_layers: GRUのlayer数
    """

    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # Embedテーブルのインスタンス化
        self.embedding = nn.Embedding(input_size, hidden_size)
        # GRUのインスタンス化
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # 入力された単語をベクトルに変換
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        # layer数分gruにベクトルを読み込ませる
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        # 隠れ層のベクトルを初期化する
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    """
    Decorderのクラス
    """

    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    """
    Attention付きのDecoderクラス
    hidden_size: 隠れ層のサイズ
    output_size: 単語数
    n_layers: GRUのレイヤー数
    dropout_p: dropoutするレート
    max_length: 出力する文の最大単語数
    """

    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        # 単語をベクトルに変換するテーブルのインスタンス化
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        #
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        #
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # デコーダーのGRUのインスタンス化
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # 隠れ層サイズのベクトルを語彙サイズのベクトルに変換する
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        """
        input: one-hotな単語ベクトル
        hidden: 隠れ層ベクトル
        encoder_output: encoderのoutputベクトル
        encoder_outputs: encoderのoutputベクトルのリスト
        """
        # 単語をベクトル化
        embedded = self.embedding(input).view(1, 1, -1)
        # 単語ベクトルに対してdropoutをかける
        embedded = self.dropout(embedded)
        # 単語ベクトルと隠れ層ベクトルをconcatして、最大単語数のベクトル（今回は20）サイズに変換、softmax関数で正規化
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        # attentionウェイトでencoderのoutputsに対して加重平均をとり、attentionベクトルを作成する
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # 単語ベクトルとattentionベクトルをconcatする
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # concatしたベクトルを隠れ層サイズに変換
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            # relu関数で非線形化
            output = F.relu(output)
            # gruに読み込ませる
            output, hidden = self.gru(output, hidden)
        # 出力されたベクトルを語彙数サイズのベクトルに変換し、softmax関数で正規化
        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        # 隠れ層ベクトルを正規化
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

