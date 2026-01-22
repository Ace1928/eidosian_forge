from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
def blockquote_reader(self, stream):
    tokens = self.parser.parse(stream.read())
    opening_tokens = filter(lambda t: t.level == 0 and t.type == 'blockquote_open', tokens)
    closing_tokens = filter(lambda t: t.level == 0 and t.type == 'blockquote_close', tokens)
    blockquotes = list()
    for o, c in zip(opening_tokens, closing_tokens):
        opening_index = tokens.index(o)
        closing_index = tokens.index(c, opening_index)
        blockquotes.append(tokens[opening_index:closing_index + 1])
    return [MarkdownBlock(self.parser.renderer.render(block, self.parser.options, env=None)) for block in blockquotes]