import re
import warnings
from math import pi, sin, cos, log, exp
from Bio.Seq import Seq, complement, complement_rna, translate
from Bio.Data import IUPACData
from Bio.Data.CodonTable import standard_dna_table
from Bio import BiopythonDeprecationWarning
def gc_fraction(seq, ambiguous='remove'):
    """Calculate G+C percentage in seq (float between 0 and 1).

    Copes with mixed case sequences. Ambiguous Nucleotides in this context are
    those different from ATCGSW (S is G or C, and W is A or T).

    If ambiguous equals "remove" (default), will only count GCS and will only
    include ACTGSW when calculating the sequence length. Equivalent to removing
    all characters in the set BDHKMNRVXY before calculating the GC content, as
    each of these ambiguous nucleotides can either be in (A,T) or in (C,G).

    If ambiguous equals "ignore", it will treat only unambiguous nucleotides (GCS)
    as counting towards the GC percentage, but will include all ambiguous and
    unambiguous nucleotides when calculating the sequence length.

    If ambiguous equals "weighted", will use a "mean" value when counting the
    ambiguous characters, for example, G and C will be counted as 1, N and X will
    be counted as 0.5, D will be counted as 0.33 etc. See Bio.SeqUtils._gc_values
    for a full list.

    Will raise a ValueError for any other value of the ambiguous parameter.


    >>> from Bio.SeqUtils import gc_fraction
    >>> seq = "ACTG"
    >>> print(f"GC content of {seq} : {gc_fraction(seq):.2f}")
    GC content of ACTG : 0.50

    S and W are ambiguous for the purposes of calculating the GC content.

    >>> seq = "ACTGSSSS"
    >>> gc = gc_fraction(seq, "remove")
    >>> print(f"GC content of {seq} : {gc:.2f}")
    GC content of ACTGSSSS : 0.75
    >>> gc = gc_fraction(seq, "ignore")
    >>> print(f"GC content of {seq} : {gc:.2f}")
    GC content of ACTGSSSS : 0.75
    >>> gc = gc_fraction(seq, "weighted")
    >>> print(f"GC content with ambiguous counting: {gc:.2f}")
    GC content with ambiguous counting: 0.75

    Some examples with ambiguous nucleotides.

    >>> seq = "ACTGN"
    >>> gc = gc_fraction(seq, "ignore")
    >>> print(f"GC content of {seq} : {gc:.2f}")
    GC content of ACTGN : 0.40
    >>> gc = gc_fraction(seq, "weighted")
    >>> print(f"GC content with ambiguous counting: {gc:.2f}")
    GC content with ambiguous counting: 0.50
    >>> gc = gc_fraction(seq, "remove")
    >>> print(f"GC content with ambiguous removing: {gc:.2f}")
    GC content with ambiguous removing: 0.50

    Ambiguous nucleotides are also removed from the length of the sequence.

    >>> seq = "GDVV"
    >>> gc = gc_fraction(seq, "ignore")
    >>> print(f"GC content of {seq} : {gc:.2f}")
    GC content of GDVV : 0.25
    >>> gc = gc_fraction(seq, "weighted")
    >>> print(f"GC content with ambiguous counting: {gc:.4f}")
    GC content with ambiguous counting: 0.6667
    >>> gc = gc_fraction(seq, "remove")
    >>> print(f"GC content with ambiguous removing: {gc:.2f}")
    GC content with ambiguous removing: 1.00


    Note that this will return zero for an empty sequence.
    """
    if ambiguous not in ('weighted', 'remove', 'ignore'):
        raise ValueError(f"ambiguous value '{ambiguous}' not recognized")
    gc = sum((seq.count(x) for x in 'CGScgs'))
    if ambiguous == 'remove':
        length = gc + sum((seq.count(x) for x in 'ATWatw'))
    else:
        length = len(seq)
    if ambiguous == 'weighted':
        gc += sum(((seq.count(x) + seq.count(x.lower())) * _gc_values[x] for x in 'BDHKMNRVXY'))
    if length == 0:
        return 0
    return gc / length