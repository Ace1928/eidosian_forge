import re
from nltk.stem.api import StemmerI
def pre32(self, word):
    """remove length three and length two prefixes in this order"""
    if len(word) >= 6:
        for pre3 in self.p3:
            if word.startswith(pre3):
                return word[3:]
    if len(word) >= 5:
        for pre2 in self.p2:
            if word.startswith(pre2):
                return word[2:]
    return word