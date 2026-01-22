import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def score_valence(self, sentiments, text):
    if sentiments:
        sum_s = float(sum(sentiments))
        punct_emph_amplifier = self._punctuation_emphasis(sum_s, text)
        if sum_s > 0:
            sum_s += punct_emph_amplifier
        elif sum_s < 0:
            sum_s -= punct_emph_amplifier
        compound = self.constants.normalize(sum_s)
        pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)
        if pos_sum > math.fabs(neg_sum):
            pos_sum += punct_emph_amplifier
        elif pos_sum < math.fabs(neg_sum):
            neg_sum -= punct_emph_amplifier
        total = pos_sum + math.fabs(neg_sum) + neu_count
        pos = math.fabs(pos_sum / total)
        neg = math.fabs(neg_sum / total)
        neu = math.fabs(neu_count / total)
    else:
        compound = 0.0
        pos = 0.0
        neg = 0.0
        neu = 0.0
    sentiment_dict = {'neg': round(neg, 3), 'neu': round(neu, 3), 'pos': round(pos, 3), 'compound': round(compound, 4)}
    return sentiment_dict