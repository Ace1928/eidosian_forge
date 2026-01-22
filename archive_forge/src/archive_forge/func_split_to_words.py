import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as sp
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
def split_to_words(self, text):
    pieces = self._encode_as_pieces(text)
    word_start = b'\xe2\x96\x81'.decode('utf-8')
    words = []
    offset = 0
    prev_end = 0
    for i, p in enumerate(pieces):
        if p.startswith(word_start):
            if offset > prev_end:
                words.append(text[prev_end:offset])
            prev_end = offset
            w = p.replace(word_start, '')
        else:
            w = p
        try:
            s = text.index(w, offset)
            pn = ''
            k = i + 1
            while k < len(pieces):
                pn = pieces[k].replace(word_start, '')
                if len(pn) > 0:
                    break
                k += 1
            if len(pn) > 0 and pn in text[offset:s]:
                offset = offset + 1
            else:
                offset = s + len(w)
        except Exception:
            offset = offset + 1
    if prev_end < offset:
        words.append(text[prev_end:offset])
    return words