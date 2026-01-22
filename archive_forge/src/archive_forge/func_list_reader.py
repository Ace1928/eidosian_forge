from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
def list_reader(self, stream):
    tokens = self.parser.parse(stream.read())
    opening_types = ('bullet_list_open', 'ordered_list_open')
    opening_tokens = filter(lambda t: t.level == 0 and t.type in opening_types, tokens)
    closing_types = ('bullet_list_close', 'ordered_list_close')
    closing_tokens = filter(lambda t: t.level == 0 and t.type in closing_types, tokens)
    list_blocks = list()
    for o, c in zip(opening_tokens, closing_tokens):
        opening_index = tokens.index(o)
        closing_index = tokens.index(c, opening_index)
        list_blocks.append(tokens[opening_index:closing_index + 1])
    return [List(tokens[0].type == 'ordered_list_open', [t.content for t in tokens if t.content]) for tokens in list_blocks]