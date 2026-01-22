import re
from collections import defaultdict, namedtuple
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.wordlist import WordListCorpusReader
from nltk.tokenize import line_tokenize
def language_codes(self):
    return self._languages.keys()