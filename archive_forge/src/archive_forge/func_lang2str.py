import re
from warnings import warn
from xml.etree import ElementTree as et
from nltk.corpus.reader import CorpusReader
def lang2str(self, lg_record):
    """Concatenate subtag values"""
    name = f'{lg_record['language']}'
    for label in ['extlang', 'script', 'region', 'variant', 'extension']:
        if label in lg_record:
            name += f': {lg_record[label]}'
    return name