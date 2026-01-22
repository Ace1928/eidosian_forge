from typing import Dict, List, Tuple
from ...pipeline import Lemmatizer
from ...tokens import Token
def unponc(word: str) -> str:
    PONC = {'ḃ': 'bh', 'ċ': 'ch', 'ḋ': 'dh', 'ḟ': 'fh', 'ġ': 'gh', 'ṁ': 'mh', 'ṗ': 'ph', 'ṡ': 'sh', 'ṫ': 'th', 'Ḃ': 'BH', 'Ċ': 'CH', 'Ḋ': 'DH', 'Ḟ': 'FH', 'Ġ': 'GH', 'Ṁ': 'MH', 'Ṗ': 'PH', 'Ṡ': 'SH', 'Ṫ': 'TH'}
    buf = []
    for ch in word:
        if ch in PONC:
            buf.append(PONC[ch])
        else:
            buf.append(ch)
    return ''.join(buf)