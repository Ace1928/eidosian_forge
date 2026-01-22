import os
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from . import Des
from . import Cla
from . import Hie
from . import Residues
from Bio import SeqIO
from Bio.Seq import Seq
def getDomainBySid(self, sid):
    """Return a domain from its sid."""
    if sid in self._sidDict:
        return self._sidDict[sid]
    if self.db_handle:
        self.getDomainFromSQL(sid=sid)
        if sid in self._sidDict:
            return self._sidDict[sid]
    else:
        return None