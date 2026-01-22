import re
from nltk.stem.api import StemmerI
def pro_w53(self, word):
    """process length five patterns and extract length three roots"""
    if word[2] in self.pr53[0] and word[0] == 'ا':
        word = word[1] + word[3:]
    elif word[3] in self.pr53[1] and word[0] == 'م':
        word = word[1:3] + word[4]
    elif word[0] in self.pr53[2] and word[4] == 'ة':
        word = word[1:4]
    elif word[0] in self.pr53[3] and word[2] == 'ت':
        word = word[1] + word[3:]
    elif word[0] in self.pr53[4] and word[2] == 'ا':
        word = word[1] + word[3:]
    elif word[2] in self.pr53[5] and word[4] == 'ة':
        word = word[:2] + word[3]
    elif word[0] in self.pr53[6] and word[1] == 'ن':
        word = word[2:]
    elif word[3] == 'ا' and word[0] == 'ا':
        word = word[1:3] + word[4]
    elif word[4] == 'ن' and word[3] == 'ا':
        word = word[:3]
    elif word[3] == 'ي' and word[0] == 'ت':
        word = word[1:3] + word[4]
    elif word[3] == 'و' and word[1] == 'ا':
        word = word[0] + word[2] + word[4]
    elif word[2] == 'ا' and word[1] == 'و':
        word = word[0] + word[3:]
    elif word[3] == 'ئ' and word[2] == 'ا':
        word = word[:2] + word[4]
    elif word[4] == 'ة' and word[1] == 'ا':
        word = word[0] + word[2:4]
    elif word[4] == 'ي' and word[2] == 'ا':
        word = word[:2] + word[3]
    else:
        word = self.suf1(word)
        if len(word) == 5:
            word = self.pre1(word)
    return word