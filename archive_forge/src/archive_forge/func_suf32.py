import re
from nltk.stem.api import StemmerI
def suf32(self, word):
    """remove length three and length two suffixes in this order"""
    if len(word) >= 6:
        for suf3 in self.s3:
            if word.endswith(suf3):
                return word[:-3]
    if len(word) >= 5:
        for suf2 in self.s2:
            if word.endswith(suf2):
                return word[:-2]
    return word