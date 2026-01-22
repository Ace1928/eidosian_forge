from abc import ABC
from abc import abstractmethod
from typing import Optional
from Bio import StreamModeError
from Bio.Align import AlignmentsAbstractBaseClass
def write_single_alignment(self, stream, alignments):
    """Write a single alignment to the output file, and return 1.

        alignments - A list or iterator returning Alignment objects
        stream     - Output file stream.
        """
    count = 0
    for alignment in alignments:
        if count == 1:
            raise ValueError(f'Alignment files in the {self.fmt} format can contain a single alignment only.')
        line = self.format_alignment(alignment)
        stream.write(line)
        count += 1
    return count