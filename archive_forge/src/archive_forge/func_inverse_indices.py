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
@property
def inverse_indices(self):
    """Return the alignment column index for each letter in each sequence.

        This property returns a list of 1D NumPy arrays; the number of arrays
        is equal to the number of aligned sequences, and the length of each
        array is equal to the length of the corresponding sequence. For each
        letter in each sequence, the array contains the corresponding column
        index in the alignment. Letters not included in the alignment are
        indicated by -1.

        For example,

        >>> from Bio import Align
        >>> aligner = Align.PairwiseAligner()
        >>> aligner.mode = "local"

        >>> alignments = aligner.align("GAACTGG", "AATG")
        >>> alignment = alignments[0]
        >>> print(alignment)
        target            1 AACTG 6
                          0 ||-|| 5
        query             0 AA-TG 4
        <BLANKLINE>
        >>> alignment.inverse_indices
        [array([-1,  0,  1,  2,  3,  4, -1]), array([0, 1, 3, 4])]
        >>> alignment = alignments[1]
        >>> print(alignment)
        target            1 AACTGG 7
                          0 ||-|-| 6
        query             0 AA-T-G 4
        <BLANKLINE>
        >>> alignment.inverse_indices
        [array([-1,  0,  1,  2,  3,  4,  5]), array([0, 1, 3, 5])]
        >>> alignments = aligner.align("GAACTGG", "CATT", strand="-")
        >>> alignment = alignments[0]
        >>> print(alignment)
        target            1 AACTG 6
                          0 ||-|| 5
        query             4 AA-TG 0
        <BLANKLINE>
        >>> alignment.inverse_indices
        [array([-1,  0,  1,  2,  3,  4, -1]), array([4, 3, 1, 0])]
        >>> alignment = alignments[1]
        >>> print(alignment)
        target            1 AACTGG 7
                          0 ||-|-| 6
        query             4 AA-T-G 0
        <BLANKLINE>
        >>> alignment.inverse_indices
        [array([-1,  0,  1,  2,  3,  4,  5]), array([5, 3, 1, 0])]

        """
    a = [-np.ones(len(sequence), int) for sequence in self.sequences]
    n, m = self.coordinates.shape
    steps = np.diff(self.coordinates, 1)
    aligned = sum(steps != 0, 0) > 1
    steps = steps[:, aligned]
    rcs = np.zeros(n, bool)
    for i, row in enumerate(steps):
        if (row >= 0).all():
            rcs[i] = False
        elif (row <= 0).all():
            rcs[i] = True
        else:
            raise ValueError(f'Inconsistent steps in row {i}')
    i = 0
    j = 0
    for k in range(m - 1):
        starts = self.coordinates[:, k]
        ends = self.coordinates[:, k + 1]
        for row, start, end, rc in zip(a, starts, ends, rcs):
            if rc == False and start < end:
                j = i + end - start
                row[start:end] = range(i, j)
            elif rc == True and start > end:
                j = i + start - end
                if end > 0:
                    row[start - 1:end - 1:-1] = range(i, j)
                elif start > 0:
                    row[start - 1::-1] = range(i, j)
        i = j
    return a