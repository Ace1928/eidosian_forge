from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
def toSeq(self):
    """Convert DNA to seq object."""
    return Seq(str(self))