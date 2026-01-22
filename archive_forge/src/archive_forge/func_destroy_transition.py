import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def destroy_transition(self, from_state, to_state):
    """Restrict transitions between the two states.

        Raises:
        KeyError if the transition is not currently allowed.

        """
    try:
        del self.transition_prob[from_state, to_state]
        del self.transition_pseudo[from_state, to_state]
    except KeyError:
        raise KeyError(f'Transition from {from_state} to {to_state} is already disallowed.')