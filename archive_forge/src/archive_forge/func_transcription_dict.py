import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def transcription_dict(self):
    """
        :return: A dictionary giving the 'standard' transcription for
            each word.
        """
    _transcriptions = {}
    with self.open('timitdic.txt') as fp:
        for line in fp:
            if not line.strip() or line[0] == ';':
                continue
            m = re.match('\\s*(\\S+)\\s+/(.*)/\\s*$', line)
            if not m:
                raise ValueError('Bad line: %r' % line)
            _transcriptions[m.group(1)] = m.group(2).split()
    return _transcriptions