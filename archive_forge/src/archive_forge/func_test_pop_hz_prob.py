import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def test_pop_hz_prob(self, fname, ext, enum_test=False, dememorization=10000, batches=20, iterations=5000):
    """Use Hardy-Weinberg test based on probability.

        Returns 2 iterators and a final tuple:

         1. Returns a loci iterator containing:
             - A dictionary[pop_pos]=(P-val, SE, Fis-WC, Fis-RH, steps).
               Some pops have a None if the info is not available.
               SE might be none (for enumerations).
             - Result of Fisher's test (Chi2, deg freedom, prob).
         2. Returns a population iterator containing:
             - A dictionary[locus]=(P-val, SE, Fis-WC, Fis-RH, steps).
               Some loci have a None if the info is not available.
               SE might be none (for enumerations).
             - Result of Fisher's test (Chi2, deg freedom, prob).
         3. Final tuple (Chi2, deg freedom, prob).

        """
    opts = self._get_opts(dememorization, batches, iterations, enum_test)
    self._run_genepop([ext], [1, 3], fname, opts)

    def hw_prob_loci_func(self):
        return _hw_func(self.stream, True, True)

    def hw_prob_pop_func(self):
        return _hw_func(self.stream, False, True)
    shutil.copyfile(fname + '.P', fname + '.P2')
    return (_FileIterator(hw_prob_loci_func, fname + '.P'), _FileIterator(hw_prob_pop_func, fname + '.P2'))