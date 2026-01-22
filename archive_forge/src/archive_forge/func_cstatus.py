from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def cstatus(self, site, delete=(), narrow=True):
    """Summarize character.

        narrow=True:  paup-mode (a c ? --> ac; ? ? ? --> ?)
        narrow=false:           (a c ? --> a c g t -; ? ? ? --> a c g t -)
        """
    undelete = [t for t in self.taxlabels if t not in delete]
    if not undelete:
        return None
    cstatus = []
    for t in undelete:
        c = self.matrix[t][site].upper()
        if self.options.get('gapmode') == 'missing' and c == self.gap:
            c = self.missing
        if narrow and c == self.missing:
            if c not in cstatus:
                cstatus.append(c)
        else:
            cstatus.extend((b for b in self.ambiguous_values[c] if b not in cstatus))
    if self.missing in cstatus and narrow and (len(cstatus) > 1):
        cstatus = [_ for _ in cstatus if _ != self.missing]
    cstatus.sort()
    return cstatus