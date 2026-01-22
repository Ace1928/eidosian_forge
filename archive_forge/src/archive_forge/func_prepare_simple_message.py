import re
from string import Formatter
@staticmethod
def prepare_simple_message(string):
    parser = AnsiParser()
    parser.feed(string)
    tokens = parser.done()
    return ColoredMessage(tokens)