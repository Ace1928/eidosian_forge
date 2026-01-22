import re
from warnings import warn
from xml.etree import ElementTree as et
from nltk.corpus.reader import CorpusReader
def wiki_dict(self, lines):
    """Convert Wikidata list of Q-codes to a BCP-47 dictionary"""
    return {pair[1]: pair[0].split('/')[-1] for pair in [line.strip().split('\t') for line in lines]}