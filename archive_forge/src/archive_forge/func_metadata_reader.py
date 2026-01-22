from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
def metadata_reader(self, stream):
    from yaml import safe_load
    return [safe_load(t.content) for t in self.parser.parse(stream.read()) if t.type == 'front_matter']