import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def polarity_scores(self, text):
    """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.

        :note: Hashtags are not taken into consideration (e.g. #BAD is neutral). If you
            are interested in processing the text in the hashtags too, then we recommend
            preprocessing your data to remove the #, after which the hashtag text may be
            matched as if it was a normal word in the sentence.
        """
    sentitext = SentiText(text, self.constants.PUNC_LIST, self.constants.REGEX_REMOVE_PUNCTUATION)
    sentiments = []
    words_and_emoticons = sentitext.words_and_emoticons
    for item in words_and_emoticons:
        valence = 0
        i = words_and_emoticons.index(item)
        if i < len(words_and_emoticons) - 1 and item.lower() == 'kind' and (words_and_emoticons[i + 1].lower() == 'of') or item.lower() in self.constants.BOOSTER_DICT:
            sentiments.append(valence)
            continue
        sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)
    sentiments = self._but_check(words_and_emoticons, sentiments)
    return self.score_valence(sentiments, text)