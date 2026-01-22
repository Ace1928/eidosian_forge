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
def complement_rna(self, inplace=False):
    """Return the complement as an RNA sequence.

        >>> Seq("CGA").complement_rna()
        Seq('GCU')

        Any T in the sequence is treated as a U:

        >>> Seq("CGAUT").complement_rna()
        Seq('GCUAA')

        In contrast, ``complement`` returns a DNA sequence by default:

        >>> Seq("CGA").complement()
        Seq('GCT')

        The sequence is modified in-place and returned if inplace is True:

        >>> my_seq = MutableSeq("CGA")
        >>> my_seq
        MutableSeq('CGA')
        >>> my_seq.complement_rna()
        MutableSeq('GCU')
        >>> my_seq
        MutableSeq('CGA')

        >>> my_seq.complement_rna(inplace=True)
        MutableSeq('GCU')
        >>> my_seq
        MutableSeq('GCU')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``complement_rna`` is called on a ``Seq`` object with ``inplace=True``.
        """
    try:
        data = self._data.translate(_rna_complement_table)
    except UndefinedSequenceError:
        return self
    if inplace:
        if not isinstance(self._data, bytearray):
            raise TypeError('Sequence is immutable')
        self._data[:] = data
        return self
    return self.__class__(data)