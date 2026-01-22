import collections
import copy
import itertools
import random
import re
import warnings
def ladderize(self, reverse=False):
    """Sort clades in-place according to the number of terminal nodes.

        Deepest clades are last by default. Use ``reverse=True`` to sort clades
        deepest-to-shallowest.
        """
    self.root.clades.sort(key=lambda c: c.count_terminals(), reverse=reverse)
    for subclade in self.root.clades:
        subclade.ladderize(reverse=reverse)