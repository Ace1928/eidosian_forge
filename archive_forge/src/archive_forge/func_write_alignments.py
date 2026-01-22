from io import StringIO
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.SeqRecord import SeqRecord
from Bio.Nexus import Nexus
def write_alignments(self, stream, alignments):
    """Write alignments to the output file, and return the number of alignments.

        alignments - A list or iterator returning Alignment objects
        """
    count = 0
    interleave = self.interleave
    for alignment in alignments:
        self.write_alignment(alignment, stream, interleave=interleave)
        count += 1
    return count