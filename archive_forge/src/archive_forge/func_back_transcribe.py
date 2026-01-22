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
def back_transcribe(self, inplace=False):
    """Return the DNA sequence from an RNA sequence by creating a new Seq object.

        >>> from Bio.Seq import Seq
        >>> messenger_rna = Seq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG")
        >>> messenger_rna
        Seq('AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG')
        >>> messenger_rna.back_transcribe()
        Seq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')

        The sequence is modified in-place and returned if inplace is True:

        >>> sequence = MutableSeq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG")
        >>> sequence
        MutableSeq('AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG')
        >>> sequence.back_transcribe()
        MutableSeq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')
        >>> sequence
        MutableSeq('AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG')

        >>> sequence.back_transcribe(inplace=True)
        MutableSeq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')
        >>> sequence
        MutableSeq('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG')

        As ``Seq`` objects are immutable, a ``TypeError`` is raised if
        ``transcribe`` is called on a ``Seq`` object with ``inplace=True``.

        Trying to back-transcribe DNA has no effect, If you have a nucleotide
        sequence which might be DNA or RNA (or even a mixture), calling the
        back-transcribe method will ensure any U becomes T.

        Trying to back-transcribe a protein sequence will replace any U for
        Selenocysteine with T for Threonine, which is biologically meaningless.

        >>> from Bio.Seq import Seq
        >>> my_protein = Seq("MAIVMGRU")
        >>> my_protein.back_transcribe()
        Seq('MAIVMGRT')
        """
    data = self._data.replace(b'U', b'T').replace(b'u', b't')
    if inplace:
        if not isinstance(self._data, bytearray):
            raise TypeError('Sequence is immutable')
        self._data[:] = data
        return self
    return self.__class__(data)