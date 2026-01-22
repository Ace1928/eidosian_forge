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
def hashedDomainsById(self, id):
    """Get domains clustered by sequence identity in a dict."""
    if id not in self.IdDatahash:
        self.IdDatahash[id] = {}
        for d in self.domainsClusteredById(id):
            self.IdDatahash[id][d] = 1
    return self.IdDatahash[id]