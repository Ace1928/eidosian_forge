import os
import sqlite3
from nltk.corpus.reader.api import CorpusReader
def language_varieties(self, lc=None):
    """
        Return a list of PanLex language varieties.

        :param lc: ISO 639 alpha-3 code. If specified, filters returned varieties
            by this code. If unspecified, all varieties are returned.
        :return: the specified language varieties as a list of tuples. The first
            element is the language variety's seven-character uniform identifier,
            and the second element is its default name.
        :rtype: list(tuple)
        """
    if lc is None:
        return self._c.execute('SELECT uid, tt FROM lv ORDER BY uid').fetchall()
    else:
        return self._c.execute('SELECT uid, tt FROM lv WHERE lc = ? ORDER BY uid', (lc,)).fetchall()