import re
import warnings
from math import pi, sin, cos, log, exp
from Bio.Seq import Seq, complement, complement_rna, translate
from Bio.Data import IUPACData
from Bio.Data.CodonTable import standard_dna_table
from Bio import BiopythonDeprecationWarning
def nt_search(seq, subseq):
    """Search for a DNA subseq in seq, return list of [subseq, positions].

    Use ambiguous values (like N = A or T or C or G, R = A or G etc.),
    searches only on forward strand.
    """
    pattern = ''
    for nt in subseq:
        value = IUPACData.ambiguous_dna_values[nt]
        if len(value) == 1:
            pattern += value
        else:
            pattern += f'[{value}]'
    pos = -1
    result = [pattern]
    while True:
        pos += 1
        s = seq[pos:]
        m = re.search(pattern, s)
        if not m:
            break
        pos += int(m.start(0))
        result.append(pos)
    return result