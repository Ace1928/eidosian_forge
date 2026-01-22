from Bio.Seq import Seq
import re
import math
from Bio import motifs
from Bio import Align
def calculate_pseudocounts(motif):
    """Calculate pseudocounts.

    Computes the root square of the total number of sequences multiplied by
    the background nucleotide.
    """
    alphabet = motif.alphabet
    background = motif.background
    total = 0
    for i in range(motif.length):
        total += sum((motif.counts[letter][i] for letter in alphabet))
    avg_nb_instances = total / motif.length
    sq_nb_instances = math.sqrt(avg_nb_instances)
    if background:
        background = dict(background)
    else:
        background = dict.fromkeys(sorted(alphabet), 1.0)
    total = sum(background.values())
    pseudocounts = {}
    for letter in alphabet:
        background[letter] /= total
        pseudocounts[letter] = sq_nb_instances * background[letter]
    return pseudocounts