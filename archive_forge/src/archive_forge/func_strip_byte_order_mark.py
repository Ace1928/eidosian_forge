from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
@classmethod
def strip_byte_order_mark(cls, data):
    """If a byte-order mark is present, strip it and return the encoding it implies.

        :param data: Some markup.
        :return: A 2-tuple (modified data, implied encoding)
        """
    encoding = None
    if isinstance(data, str):
        return (data, encoding)
    if len(data) >= 4 and data[:2] == b'\xfe\xff' and (data[2:4] != '\x00\x00'):
        encoding = 'utf-16be'
        data = data[2:]
    elif len(data) >= 4 and data[:2] == b'\xff\xfe' and (data[2:4] != '\x00\x00'):
        encoding = 'utf-16le'
        data = data[2:]
    elif data[:3] == b'\xef\xbb\xbf':
        encoding = 'utf-8'
        data = data[3:]
    elif data[:4] == b'\x00\x00\xfe\xff':
        encoding = 'utf-32be'
        data = data[4:]
    elif data[:4] == b'\xff\xfe\x00\x00':
        encoding = 'utf-32le'
        data = data[4:]
    return (data, encoding)