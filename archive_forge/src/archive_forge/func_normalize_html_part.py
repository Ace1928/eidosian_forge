from __future__ import print_function
import re
import hashlib
@staticmethod
def normalize_html_part(s):
    data = []
    stripper = HTMLStripper(data)
    try:
        stripper.feed(s)
    except (UnicodeDecodeError, HTMLParser.HTMLParseError):
        pass
    return ' '.join(data)