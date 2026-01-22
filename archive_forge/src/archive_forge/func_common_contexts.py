import re
import sys
from collections import Counter, defaultdict, namedtuple
from functools import reduce
from math import log
from nltk.collocations import BigramCollocationFinder
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.metrics import BigramAssocMeasures, f_measure
from nltk.probability import ConditionalFreqDist as CFD
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.util import LazyConcatenation, tokenwrap
def common_contexts(self, words, num=20):
    """
        Find contexts where the specified words appear; list
        most frequent common contexts first.

        :param words: The words used to seed the similarity search
        :type words: str
        :param num: The number of words to generate (default=20)
        :type num: int
        :seealso: ContextIndex.common_contexts()
        """
    if '_word_context_index' not in self.__dict__:
        self._word_context_index = ContextIndex(self.tokens, key=lambda s: s.lower())
    try:
        fd = self._word_context_index.common_contexts(words, True)
        if not fd:
            print('No common contexts were found')
        else:
            ranked_contexts = [w for w, _ in fd.most_common(num)]
            print(tokenwrap((w1 + '_' + w2 for w1, w2 in ranked_contexts)))
    except ValueError as e:
        print(e)