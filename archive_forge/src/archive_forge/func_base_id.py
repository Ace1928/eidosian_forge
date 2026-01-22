from Bio.Seq import Seq
import re
import math
from Bio import motifs
from Bio import Align
@property
def base_id(self):
    """Return the JASPAR base matrix ID."""
    base_id, __ = split_jaspar_id(self.matrix_id)
    return base_id