import re
from string import Formatter
@staticmethod
def prepare_format(string):
    tokens, messages_color_tokens = Colorizer._parse_without_formatting(string)
    return ColoredFormat(tokens, messages_color_tokens)