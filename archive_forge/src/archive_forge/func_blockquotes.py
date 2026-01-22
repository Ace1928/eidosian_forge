from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
@comma_separated_string_args
def blockquotes(self, fileids=None, categories=None):
    return self.concatenated_view(self.blockquote_reader, fileids, categories)