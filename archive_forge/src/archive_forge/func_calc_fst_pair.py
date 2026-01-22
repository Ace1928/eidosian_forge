import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def calc_fst_pair(self, fname):
    """Estimate spatial structure from Allele identity for all population pairs."""
    self._run_genepop(['.ST2', '.MIG'], [6, 2], fname)
    with open(fname + '.ST2') as f:
        line = f.readline()
        while line != '':
            line = line.rstrip()
            if line.startswith('Estimates for all loci'):
                avg_fst = _read_headed_triangle_matrix(f)
            line = f.readline()

    def loci_func(self):
        line = self.stream.readline()
        while line != '':
            line = line.rstrip()
            m = re.search(' Locus: (.+)', line)
            if m is not None:
                locus = m.group(1)
                matrix = _read_headed_triangle_matrix(self.stream)
                return (locus, matrix)
            line = self.stream.readline()
        self.done = True
        raise StopIteration
    os.remove(fname + '.MIG')
    return (_FileIterator(loci_func, fname + '.ST2'), avg_fst)