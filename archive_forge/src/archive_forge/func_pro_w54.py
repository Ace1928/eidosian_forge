import re
from nltk.stem.api import StemmerI
def pro_w54(self, word):
    """process length five patterns and extract length four roots"""
    if word[0] in self.pr53[2]:
        word = word[1:]
    elif word[4] == 'ة':
        word = word[:4]
    elif word[2] == 'ا':
        word = word[:2] + word[3:]
    return word