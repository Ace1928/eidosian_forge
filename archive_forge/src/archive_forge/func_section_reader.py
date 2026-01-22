from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
def section_reader(self, stream):
    section_blocks, block = (list(), list())
    in_heading = False
    for t in self.parser.parse(stream.read()):
        if t.level == 0 and t.type == 'heading_open':
            if block:
                section_blocks.append(block)
            block = list()
            in_heading = True
        if in_heading:
            block.append(t)
    return [MarkdownSection(block[1].content, block[0].markup.count('#'), self.parser.renderer.render(block, self.parser.options, env=None)) for block in section_blocks]