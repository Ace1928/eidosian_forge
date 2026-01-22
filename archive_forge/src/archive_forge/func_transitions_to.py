import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def transitions_to(self, state_letter):
    """Get all source states which can transition to destination state_letter.

        This returns all letters which the given state_letter is reachable
        from, i.e. all the source states which can reach state_later

        An empty list is returned if state_letter is unreachable.
        """
    if state_letter in self._transitions_to:
        return self._transitions_to[state_letter]
    else:
        return []