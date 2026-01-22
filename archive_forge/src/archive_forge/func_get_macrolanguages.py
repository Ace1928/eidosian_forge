import re
from collections import defaultdict, namedtuple
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.wordlist import WordListCorpusReader
from nltk.tokenize import line_tokenize
def get_macrolanguages(self):
    macro_langauges = defaultdict(list)
    for lang in self._languages.values():
        macro_langauges[lang.iso639].append(lang.panlex_uid)
    return macro_langauges