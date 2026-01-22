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
def getSeq(self, domain):
    """Return seq associated with domain."""
    return self.getSeqBySid(domain.sid)