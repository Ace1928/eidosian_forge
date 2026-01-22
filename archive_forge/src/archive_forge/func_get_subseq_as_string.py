import os
from . import BioSeq
from . import Loader
from . import DBUtils
def get_subseq_as_string(self, seqid, start, end):
    """Return a substring of a sequence.

        Arguments:
         - seqid - The internal id for the sequence
         - start - The start position of the sequence; 0-indexed
         - end - The end position of the sequence

        """
    length = end - start
    return self.execute_one('SELECT SUBSTR(seq, %s, %s) FROM biosequence WHERE bioentry_id = %s', (start + 1, length, seqid))[0]