import base64
import datetime
import re
import unicodedata
from binascii import Error as BinasciiError
from email.utils import formatdate
from urllib.parse import quote, unquote
from urllib.parse import urlencode as original_urlencode
from urllib.parse import urlparse
from django.utils.datastructures import MultiValueDict
from django.utils.regex_helper import _lazy_re_compile
def parse_header_parameters(line):
    """
    Parse a Content-type like header.
    Return the main content-type and a dictionary of options.
    """
    parts = _parseparam(';' + line)
    key = parts.__next__().lower()
    pdict = {}
    for p in parts:
        i = p.find('=')
        if i >= 0:
            has_encoding = False
            name = p[:i].strip().lower()
            if name.endswith('*'):
                name = name[:-1]
                if p.count("'") == 2:
                    has_encoding = True
            value = p[i + 1:].strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
                value = value.replace('\\\\', '\\').replace('\\"', '"')
            if has_encoding:
                encoding, lang, value = value.split("'")
                value = unquote(value, encoding=encoding)
            pdict[name] = value
    return (key, pdict)