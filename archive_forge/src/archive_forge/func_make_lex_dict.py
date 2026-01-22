import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def make_lex_dict(self):
    """
        Convert lexicon file to a dictionary
        """
    lex_dict = {}
    for line in self.lexicon_file.split('\n'):
        word, measure = line.strip().split('\t')[0:2]
        lex_dict[word] = float(measure)
    return lex_dict