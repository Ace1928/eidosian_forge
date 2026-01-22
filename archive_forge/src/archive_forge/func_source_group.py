import os
import sqlite3
from nltk.corpus.reader.api import CorpusReader
def source_group(self):
    """
        :return: the meaning's source group id.
        :rtype: int
        """
    return self['ui']