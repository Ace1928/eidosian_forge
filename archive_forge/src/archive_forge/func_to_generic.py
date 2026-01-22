from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
def to_generic(self):
    """Retrieve generic alignment object for the given alignment.

        Instead of the tuples, this returns a MultipleSeqAlignment object
        from Bio.Align, through which you can manipulate and query
        the object.

        Thanks to James Casbon for the code.
        """
    seq_parts = []
    seq_names = []
    parse_number = 0
    n = 0
    for name, start, seq, end in self.alignment:
        if name == 'QUERY':
            parse_number += 1
            n = 0
        if parse_number == 1:
            seq_parts.append(seq)
            seq_names.append(name)
        else:
            seq_parts[n] += seq
            n += 1
    records = (SeqRecord(Seq(seq), name) for name, seq in zip(seq_names, seq_parts))
    return MultipleSeqAlignment(records)