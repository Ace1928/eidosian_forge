import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def iscompatible(self, other):
    """Check if current bitstr1 is compatible with another bitstr2.

        Two conditions are considered as compatible:
         1. bitstr1.contain(bitstr2) or vice versa;
         2. bitstr1.independent(bitstr2).

        """
    return self.contains(other) or other.contains(self) or self.independent(other)