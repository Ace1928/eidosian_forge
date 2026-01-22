import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def test_ld(self, fname, dememorization=10000, batches=20, iterations=5000):
    """Test for linkage disequilibrium on each pair of loci in each population."""
    opts = self._get_opts(dememorization, batches, iterations)
    self._run_genepop(['.DIS'], [2, 1], fname, opts)

    def ld_pop_func(self):
        current_pop = None
        line = self.stream.readline().rstrip()
        if line == '':
            self.done = True
            raise StopIteration
        toks = [x for x in line.split(' ') if x != '']
        pop, locus1, locus2 = (toks[0], toks[1], toks[2])
        if not hasattr(self, 'start_locus1'):
            start_locus1, start_locus2 = (locus1, locus2)
            current_pop = -1
        if locus1 == start_locus1 and locus2 == start_locus2:
            current_pop += 1
        if toks[3] == 'No':
            return (current_pop, pop, (locus1, locus2), None)
        p, se, switches = (_gp_float(toks[3]), _gp_float(toks[4]), _gp_int(toks[5]))
        return (current_pop, pop, (locus1, locus2), (p, se, switches))

    def ld_func(self):
        line = self.stream.readline().rstrip()
        if line == '':
            self.done = True
            raise StopIteration
        toks = [x for x in line.split(' ') if x != '']
        locus1, locus2 = (toks[0], toks[2])
        try:
            chi2, df, p = (_gp_float(toks[3]), _gp_int(toks[4]), _gp_float(toks[5]))
        except ValueError:
            return ((locus1, locus2), None)
        return ((locus1, locus2), (chi2, df, p))
    f1 = open(fname + '.DIS')
    line = f1.readline()
    while '----' not in line:
        line = f1.readline()
    shutil.copyfile(fname + '.DIS', fname + '.DI2')
    f2 = open(fname + '.DI2')
    line = f2.readline()
    while 'Locus pair' not in line:
        line = f2.readline()
    while '----' not in line:
        line = f2.readline()
    return (_FileIterator(ld_pop_func, fname + '.DIS', f1), _FileIterator(ld_func, fname + '.DI2', f2))