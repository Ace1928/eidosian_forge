import os
from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import add, and_
from nltk.data import show_cfg
from nltk.inference.mace import MaceCommand
from nltk.inference.prover9 import Prover9Command
from nltk.parse import load_parser
from nltk.parse.malt import MaltParser
from nltk.sem.drt import AnaphoraResolutionException, resolve_anaphora
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Expression
from nltk.tag import RegexpTagger
def readings(self, sentence=None, threaded=False, verbose=True, filter=False, show_thread_readings=False):
    """
        Construct and show the readings of the discourse (or of a single sentence).

        :param sentence: test just this sentence
        :type sentence: str
        :param threaded: if ``True``, print out each thread ID and the corresponding thread.
        :param filter: if ``True``, only print out consistent thread IDs and threads.
        """
    self._construct_readings()
    self._construct_threads()
    if filter or show_thread_readings:
        threaded = True
    if verbose:
        if not threaded:
            self._show_readings(sentence=sentence)
        else:
            self._show_threads(filter=filter, show_thread_readings=show_thread_readings)