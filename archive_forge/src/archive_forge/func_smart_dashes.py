from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def smart_dashes(chars):
    en_count = 0
    em_count = 0
    if len(chars) % 3 == 0:
        em_count = len(chars) // 3
    elif len(chars) % 2 == 0:
        en_count = len(chars) // 2
    elif len(chars) % 3 == 2:
        en_count = 1
        em_count = (len(chars) - 2) // 3
    else:
        en_count = 2
        em_count = (len(chars) - 4) // 3
    return '—' * em_count + '–' * en_count