import re
from os import path
from nltk.corpus.reader import CorpusReader
from nltk.data import ZipFilePathPointer
from nltk.probability import FreqDist
def iso_to_crubadan(self, lang):
    """Return internal Crubadan code based on ISO 639-3 code"""
    for i in self._lang_mapping_data:
        if i[1].lower() == lang.lower():
            return i[0]