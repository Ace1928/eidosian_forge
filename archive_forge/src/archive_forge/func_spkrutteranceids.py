import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def spkrutteranceids(self, speaker):
    """
        :return: A list of all utterances associated with a given
            speaker.
        """
    return [utterance for utterance in self._utterances if utterance.startswith(speaker + '/')]