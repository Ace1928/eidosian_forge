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
def getNodeBySunid(self, sunid):
    """Return a node from its sunid."""
    if sunid in self._sunidDict:
        return self._sunidDict[sunid]
    if self.db_handle:
        self.getDomainFromSQL(sunid=sunid)
        if sunid in self._sunidDict:
            return self._sunidDict[sunid]
    else:
        return None