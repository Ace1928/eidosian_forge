import os
from . import BioSeq
from . import Loader
from . import DBUtils
def fetch_seqid_by_identifier(self, dbid, identifier):
    """Return the internal id for a sequence using its identifier.

        Arguments:
         - dbid - the internal id for the sub-database
         - identifier - the identifier of the sequence. Corresponds to
           the identifier column of the bioentry table in the SQL schema.

        """
    sql = 'SELECT bioentry_id FROM bioentry WHERE identifier = %s'
    fields = [identifier]
    if dbid:
        sql += ' and biodatabase_id = %s'
        fields.append(dbid)
    self.execute(sql, fields)
    rv = self.cursor.fetchall()
    if not rv:
        raise IndexError(f'Cannot find display id {identifier!r}')
    return rv[0][0]