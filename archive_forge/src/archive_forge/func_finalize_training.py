import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def finalize_training(self, verbose=False):
    """
        Uses data that has been gathered in training to determine likely
        collocations and sentence starters.
        """
    self._params.clear_sent_starters()
    for typ, log_likelihood in self._find_sent_starters():
        self._params.sent_starters.add(typ)
        if verbose:
            print(f'  Sent Starter: [{log_likelihood:6.4f}] {typ!r}')
    self._params.clear_collocations()
    for (typ1, typ2), log_likelihood in self._find_collocations():
        self._params.collocations.add((typ1, typ2))
        if verbose:
            print(f'  Collocation: [{log_likelihood:6.4f}] {typ1!r}+{typ2!r}')
    self._finalized = True