import re
from nltk.stem.api import StemmerI
def end_w5(self, word):
    """ending step (word of length five)"""
    if len(word) == 4:
        word = self.pro_w4(word)
    elif len(word) == 5:
        word = self.pro_w54(word)
    return word