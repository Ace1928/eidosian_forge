from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq, reverse_complement
from Bio.SeqRecord import SeqRecord
def format_alignment(self, alignment):
    """Return a string with a single alignment in the Mauve format."""
    metadata = self._metadata
    n, m = alignment.shape
    if n == 0:
        raise ValueError('Must have at least one sequence')
    if m == 0:
        raise ValueError('Non-empty sequences are required')
    filename = metadata.get('File')
    lines = []
    for i in range(n):
        identifier = alignment.sequences[i].id
        start = alignment.coordinates[i, 0]
        end = alignment.coordinates[i, -1]
        if start <= end:
            strand = '+'
        else:
            strand = '-'
            start, end = (end, start)
        if start == end:
            assert start == 0
        else:
            start += 1
        sequence = alignment[i]
        if filename is None:
            number = self._identifiers.index(identifier) + 1
            line = f'> {number}:{start}-{end} {strand} {identifier}\n'
        else:
            number = int(identifier) + 1
            line = f'> {number}:{start}-{end} {strand} {filename}\n'
        lines.append(line)
        line = f'{sequence}\n'
        lines.append(line)
    lines.append('=\n')
    return ''.join(lines)