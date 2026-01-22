import os
from random import choice
from itertools import combinations
import snappy
from plink import LinkManager
from .twister_core import build_bundle, build_splitting, twister_version
def random_word(self, n, twists=True, half_twists=True, macros=True, inverses=True):
    """ Returns a random word of length n in the generators of Mod(S). Setting twists, half_twists or macros to False
		prevents them from appearing in the word. If all generators are disallowed then the empty word is returned. If 
		inverses is set to False then no inverse of a generator will be used. """
    generators = []
    if twists:
        generators += [curve[0] for curve in self.curves['annulus']]
    if twists and inverses:
        generators += [curve[1] for curve in self.curves['annulus']]
    if half_twists:
        generators += [curve[0] for curve in self.curves['rectangle']]
    if half_twists and inverses:
        generators += [curve[1] for curve in self.curves['rectangle']]
    if macros:
        generators += [curve[0] for curve in self.curves['macro']]
    if generators == []:
        return ''
    return ''.join([choice(generators) for i in range(n)])