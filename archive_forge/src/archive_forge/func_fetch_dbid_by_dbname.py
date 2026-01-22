import os
from . import BioSeq
from . import Loader
from . import DBUtils
def fetch_dbid_by_dbname(self, dbname):
    """Return the internal id for the sub-database using its name."""
    self.execute('select biodatabase_id from biodatabase where name = %s', (dbname,))
    rv = self.cursor.fetchall()
    if not rv:
        raise KeyError(f'Cannot find biodatabase with name {dbname!r}')
    return rv[0][0]