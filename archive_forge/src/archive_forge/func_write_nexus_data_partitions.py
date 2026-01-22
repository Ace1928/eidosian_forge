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
def write_nexus_data_partitions(self, matrix=None, filename=None, blocksize=None, interleave=False, exclude=(), delete=(), charpartition=None, comment='', mrbayes=False):
    """Write a nexus file for each partition in charpartition.

        Only non-excluded characters and non-deleted taxa are included,
        just the data block is written.
        """
    if not matrix:
        matrix = self.matrix
    if not matrix:
        return
    if not filename:
        filename = self.filename
    if charpartition:
        pfilenames = {}
        for p in charpartition:
            total_exclude = list(exclude)
            total_exclude.extend((c for c in range(self.nchar) if c not in charpartition[p]))
            total_exclude = _make_unique(total_exclude)
            pcomment = comment + '\nPartition: ' + p + '\n'
            dot = filename.rfind('.')
            if dot > 0:
                pfilename = filename[:dot] + '_' + p + '.data'
            else:
                pfilename = filename + '_' + p
            pfilenames[p] = pfilename
            self.write_nexus_data(filename=pfilename, matrix=matrix, blocksize=blocksize, interleave=interleave, exclude=total_exclude, delete=delete, comment=pcomment, append_sets=False, mrbayes=mrbayes)
        return pfilenames
    else:
        fn = self.filename + '.data'
        self.write_nexus_data(filename=fn, matrix=matrix, blocksize=blocksize, interleave=interleave, exclude=exclude, delete=delete, comment=comment, append_sets=False, mrbayes=mrbayes)
        return fn