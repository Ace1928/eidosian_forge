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
def export_fasta(self, filename=None, width=70):
    """Write matrix into a fasta file."""
    if not filename:
        if '.' in self.filename and self.filename.split('.')[-1].lower() in ['paup', 'nexus', 'nex', 'dat']:
            filename = '.'.join(self.filename.split('.')[:-1]) + '.fas'
        else:
            filename = self.filename + '.fas'
    with open(filename, 'w') as fh:
        for taxon in self.taxlabels:
            fh.write('>' + safename(taxon) + '\n')
            for i in range(0, len(str(self.matrix[taxon])), width):
                fh.write(str(self.matrix[taxon])[i:i + width] + '\n')
    return filename