import os
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from . import Des
from . import Cla
from . import Hie
from . import Residues
from Bio import SeqIO
from Bio.Seq import Seq
def cmp_sccs(sccs1, sccs2):
    """Order SCOP concise classification strings (sccs).

    a.4.5.1 < a.4.5.11 < b.1.1.1

    A sccs (e.g. a.4.5.11) compactly represents a domain's classification.
    The letter represents the class, and the numbers are the fold,
    superfamily, and family, respectively.
    """
    s1 = sccs1.split('.')
    s2 = sccs2.split('.')
    c1, c2 = (s1[0], s2[0])
    if c1 < c2:
        return -1
    if c1 > c2:
        return +1
    for c1, c2 in zip(s1[1:], s2[1:]):
        i1 = int(c1)
        i2 = int(c2)
        if i1 < i2:
            return -1
        if i1 > i2:
            return +1
    n1 = len(s1)
    n2 = len(s2)
    if n1 < n2:
        return -1
    if n1 > n2:
        return +1
    return 0