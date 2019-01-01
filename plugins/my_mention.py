# coding: utf-8
"""
自動返信のメイン部分
"""
import torch
from filer3 import Filer
from slackbot.bot import default_reply  # 該当する応答がない場合に反応するデコーダ

from plugins.seq2seq import EncoderRNN, AttnDecoderRNN, reply_message, Lang


hidden_size = 256

input_lang = Filer.read_pkl('./plugins/file/input_lang.pkl')
output_lang = Filer.read_pkl('./plugins/file/output_lang.pkl')

encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               1, dropout_p=0.1)

encoder1.load_state_dict(torch.load('./plugins/model/encoder_75000.model'))
attn_decoder1.load_state_dict(
    torch.load('./plugins/model/decoder_75000.model')
)

print('models are loaded!')


@default_reply()
def default_func(message):
    text = message.body['text']     # メッセージを取り出す
    # 送信メッセージを作る。改行やトリプルバッククォートで囲む表現も可能

    msg = reply_message(text, encoder1, attn_decoder1, input_lang, output_lang)
    message.reply(msg)      # メンション
