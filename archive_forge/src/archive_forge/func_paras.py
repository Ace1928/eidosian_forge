import nltk.data
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
def paras(self, fileids=None):
    raise NotImplementedError('The Europarl corpus reader does not support paragraphs. Please use chapters() instead.')