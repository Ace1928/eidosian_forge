from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def train_supervised(self, labelled_sequences, estimator=None):
    """
        Supervised training maximising the joint probability of the symbol and
        state sequences. This is done via collecting frequencies of
        transitions between states, symbol observations while within each
        state and which states start a sentence. These frequency distributions
        are then normalised into probability estimates, which can be
        smoothed if desired.

        :return: the trained model
        :rtype: HiddenMarkovModelTagger
        :param labelled_sequences: the training data, a set of
            labelled sequences of observations
        :type labelled_sequences: list
        :param estimator: a function taking
            a FreqDist and a number of bins and returning a CProbDistI;
            otherwise a MLE estimate is used
        """
    if estimator is None:
        estimator = lambda fdist, bins: MLEProbDist(fdist)
    known_symbols = set(self._symbols)
    known_states = set(self._states)
    starting = FreqDist()
    transitions = ConditionalFreqDist()
    outputs = ConditionalFreqDist()
    for sequence in labelled_sequences:
        lasts = None
        for token in sequence:
            state = token[_TAG]
            symbol = token[_TEXT]
            if lasts is None:
                starting[state] += 1
            else:
                transitions[lasts][state] += 1
            outputs[state][symbol] += 1
            lasts = state
            if state not in known_states:
                self._states.append(state)
                known_states.add(state)
            if symbol not in known_symbols:
                self._symbols.append(symbol)
                known_symbols.add(symbol)
    N = len(self._states)
    pi = estimator(starting, N)
    A = ConditionalProbDist(transitions, estimator, N)
    B = ConditionalProbDist(outputs, estimator, len(self._symbols))
    return HiddenMarkovModelTagger(self._symbols, self._states, A, B, pi)