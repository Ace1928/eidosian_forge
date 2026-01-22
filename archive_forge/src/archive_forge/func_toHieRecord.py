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
def toHieRecord(self):
    """Return an Hie.Record."""
    rec = Hie.Record()
    rec.sunid = str(self.sunid)
    if self.getParent():
        rec.parent = str(self.getParent().sunid)
    else:
        rec.parent = '-'
    for c in self.getChildren():
        rec.children.append(str(c.sunid))
    return rec