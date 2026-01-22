import os
from . import BioSeq
from . import Loader
from . import DBUtils
def list_bioentry_ids(self, dbid):
    """Return a list of internal ids for all of the sequences in a sub-databae.

        Arguments:
         - dbid - The internal id for a sub-database

        """
    return self.execute_and_fetch_col0('SELECT bioentry_id FROM bioentry WHERE biodatabase_id = %s', (dbid,))