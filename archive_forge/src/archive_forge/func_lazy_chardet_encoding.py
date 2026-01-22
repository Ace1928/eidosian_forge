import codecs
import re
import typing as t
from .exceptions import (
def lazy_chardet_encoding(data):
    return chardet.detect(data)['encoding'] or ''