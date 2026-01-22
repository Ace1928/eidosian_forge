import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
def calc_affine_penalty(length, open, extend, penalize_extend_when_opening):
    """Calculate a penalty score for the gap function."""
    if length <= 0:
        return 0.0
    penalty = open + extend * length
    if not penalize_extend_when_opening:
        penalty -= extend
    return penalty