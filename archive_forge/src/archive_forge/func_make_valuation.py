import os
import re
import shelve
import sys
import nltk.data
def make_valuation(concepts, read=False, lexicon=False):
    """
    Convert a list of ``Concept`` objects into a list of (label, extension) pairs;
    optionally create a ``Valuation`` object.

    :param concepts: concepts
    :type concepts: list(Concept)
    :param read: if ``True``, ``(symbol, set)`` pairs are read into a ``Valuation``
    :type read: bool
    :rtype: list or Valuation
    """
    vals = []
    for c in concepts:
        vals.append((c.prefLabel, c.extension))
    if lexicon:
        read = True
    if read:
        from nltk.sem import Valuation
        val = Valuation({})
        val.update(vals)
        val = label_indivs(val, lexicon=lexicon)
        return val
    else:
        return vals