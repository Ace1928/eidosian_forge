import random
import re
import textwrap
import time
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from nltk.chunk import ChunkScore, RegexpChunkParser
from nltk.chunk.regexp import RegexpChunkRule
from nltk.corpus import conll2000, treebank_chunk
from nltk.draw.util import ShowText
from nltk.tree import Tree
from nltk.util import in_idle
def normalize_grammar(self, grammar):
    grammar = re.sub('((\\\\.|[^#])*)(#.*)?', '\\1', grammar)
    grammar = re.sub(' +', ' ', grammar)
    grammar = re.sub('\\n\\s+', '\\n', grammar)
    grammar = grammar.strip()
    grammar = re.sub('([^\\\\])\\$', '\\1\\\\$', grammar)
    return grammar