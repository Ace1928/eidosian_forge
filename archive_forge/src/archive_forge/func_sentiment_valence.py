import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def sentiment_valence(self, valence, sentitext, item, i, sentiments):
    is_cap_diff = sentitext.is_cap_diff
    words_and_emoticons = sentitext.words_and_emoticons
    item_lowercase = item.lower()
    if item_lowercase in self.lexicon:
        valence = self.lexicon[item_lowercase]
        if item.isupper() and is_cap_diff:
            if valence > 0:
                valence += self.constants.C_INCR
            else:
                valence -= self.constants.C_INCR
        for start_i in range(0, 3):
            if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                s = self.constants.scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                if start_i == 1 and s != 0:
                    s = s * 0.95
                if start_i == 2 and s != 0:
                    s = s * 0.9
                valence = valence + s
                valence = self._never_check(valence, words_and_emoticons, start_i, i)
                if start_i == 2:
                    valence = self._idioms_check(valence, words_and_emoticons, i)
        valence = self._least_check(valence, words_and_emoticons, i)
    sentiments.append(valence)
    return sentiments