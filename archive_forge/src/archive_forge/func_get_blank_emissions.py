import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def get_blank_emissions(self):
    """Get the starting default emmissions for each sequence.

        This returns a dictionary of the default emmissions for each
        letter. The dictionary is structured with keys as
        (seq_letter, emmission_letter) and values as the starting number
        of emmissions.
        """
    return self._emission_pseudo