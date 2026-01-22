import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def set_emission_score(self, seq_state, emission_state, probability):
    """Set the probability of a emission from a particular state.

        Raises:
        KeyError if the emission from the given state is not allowed.

        """
    if (seq_state, emission_state) in self.emission_prob:
        self.emission_prob[seq_state, emission_state] = probability
    else:
        raise KeyError(f'Emission of {emission_state} from {seq_state} is not allowed.')