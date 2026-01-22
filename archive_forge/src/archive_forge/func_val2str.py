import re
from warnings import warn
from xml.etree import ElementTree as et
from nltk.corpus.reader import CorpusReader
def val2str(self, val):
    """Return only first value"""
    if type(val) == list:
        val = val[0]
    return val