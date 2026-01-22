import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def utteranceids(self, dialect=None, sex=None, spkrid=None, sent_type=None, sentid=None):
    """
        :return: A list of the utterance identifiers for all
            utterances in this corpus, or for the given speaker, dialect
            region, gender, sentence type, or sentence number, if
            specified.
        """
    if isinstance(dialect, str):
        dialect = [dialect]
    if isinstance(sex, str):
        sex = [sex]
    if isinstance(spkrid, str):
        spkrid = [spkrid]
    if isinstance(sent_type, str):
        sent_type = [sent_type]
    if isinstance(sentid, str):
        sentid = [sentid]
    utterances = self._utterances[:]
    if dialect is not None:
        utterances = [u for u in utterances if u[2] in dialect]
    if sex is not None:
        utterances = [u for u in utterances if u[4] in sex]
    if spkrid is not None:
        utterances = [u for u in utterances if u[:9] in spkrid]
    if sent_type is not None:
        utterances = [u for u in utterances if u[11] in sent_type]
    if sentid is not None:
        utterances = [u for u in utterances if u[10:] in spkrid]
    return utterances