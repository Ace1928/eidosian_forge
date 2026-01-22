import array
import collections
import numbers
import warnings
from abc import ABC
from abc import abstractmethod
from typing import overload, Optional, Union, Dict
from Bio import BiopythonWarning
from Bio.Data import CodonTable
from Bio.Data import IUPACData
def reverse_complement_rna(self, inplace=False):
    """Return the reverse complement as an RNA sequence.

        >>> Seq("CGA").reverse_complement_rna()
        Seq('UCG')

        Any T in the sequence is treated as a U:

        >>> Seq("CGAUT").reverse_complement_rna()
        Seq('AAUCG')

        In contrast, ``reverse_complement`` returns a DNA sequence:

        >>> Seq("CGA").reverse_complement()
        Seq('TCG')

        The sequence is modified in-place and returned if inplace is True:

        >>> my_seq = MutableSeq("CGA")
        >>> my_seq
        MutableSeq('CGA')
        >>> my_seq.reverse_complement_rna()
        MutableSeq('UCG')
        >>> my_seq
        MutableSeq('CGA')

        >>> my_seq.reverse_complement_rna(inplace=True)
        MutableSeq('UCG')
        >>> my_seq
        MutableSeq('UCG')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``reverse_complement_rna`` is called on a ``Seq`` object with
        ``inplace=True``.
        """
    try:
        data = self._data.translate(_rna_complement_table)
    except UndefinedSequenceError:
        return self
    if inplace:
        if not isinstance(self._data, bytearray):
            raise TypeError('Sequence is immutable')
        self._data[::-1] = data
        return self
    return self.__class__(data[::-1])