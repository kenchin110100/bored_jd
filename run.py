# coding: utf-8
"""
slackbotのメインのファイル
"""
from slackbot.bot import Bot
from plugins.seq2seq import Lang


def main():
    bot = Bot()
    bot.run()


if __name__ == "__main__":
    print('start slackbot')
    main()
