import sys
import collections
import copy
import importlib
import types
import warnings
import numbers
from itertools import zip_longest
from abc import ABC, abstractmethod
from typing import Dict
from Bio.Align import _pairwisealigner  # type: ignore
from Bio.Align import _codonaligner  # type: ignore
from Bio.Align import substitution_matrices
from Bio.Data import CodonTable
from Bio.Seq import Seq, MutableSeq, reverse_complement, UndefinedSequenceError
from Bio.Seq import translate
from Bio.SeqRecord import SeqRecord, _RestrictedDict
def get_alignment_length(self):
    """Return the maximum length of the alignment.

        All objects in the alignment should (hopefully) have the same
        length. This function will go through and find this length
        by finding the maximum length of sequences in the alignment.

        >>> from Bio.Seq import Seq
        >>> from Bio.SeqRecord import SeqRecord
        >>> from Bio.Align import MultipleSeqAlignment
        >>> a = SeqRecord(Seq("ACTGCTAGCTAG"), id="Alpha")
        >>> b = SeqRecord(Seq("ACT-CTAGCTAG"), id="Beta")
        >>> c = SeqRecord(Seq("ACTGCTAGATAG"), id="Gamma")
        >>> align = MultipleSeqAlignment([a, b, c])
        >>> align.get_alignment_length()
        12

        If you want to know the number of sequences in the alignment,
        use len(align) instead:

        >>> len(align)
        3

        """
    max_length = 0
    for record in self._records:
        if len(record.seq) > max_length:
            max_length = len(record.seq)
    return max_length