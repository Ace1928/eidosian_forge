import re
from collections import defaultdict
from functools import reduce
from nltk.corpus.reader import CorpusReader

        Determines whether or not the given ngram is in the thesaurus.

        :param ngram: ngram to lookup
        :type ngram: C{string}
        :return: whether the given ngram is in the thesaurus.
        