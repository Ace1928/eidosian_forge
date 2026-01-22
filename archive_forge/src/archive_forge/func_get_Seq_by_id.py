import os
from . import BioSeq
from . import Loader
from . import DBUtils
def get_Seq_by_id(self, name):
    """Get a DBSeqRecord object by its name.

        Example: seq_rec = db.get_Seq_by_id('ROA1_HUMAN')

        The name of this method is misleading since it returns a DBSeqRecord
        rather than a Seq object, and presumably was to mirror BioPerl.
        """
    seqid = self.adaptor.fetch_seqid_by_display_id(self.dbid, name)
    return BioSeq.DBSeqRecord(self.adaptor, seqid)