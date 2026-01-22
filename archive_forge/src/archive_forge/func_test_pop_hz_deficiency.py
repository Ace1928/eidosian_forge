import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def test_pop_hz_deficiency(self, fname, enum_test=True, dememorization=10000, batches=20, iterations=5000):
    """Use Hardy-Weinberg test for heterozygote deficiency.

        Returns a population iterator containing a dictionary wehre
        dictionary[locus]=(P-val, SE, Fis-WC, Fis-RH, steps).

        Some loci have a None if the info is not available.
        SE might be none (for enumerations).
        """
    return self._test_pop_hz_both(fname, 1, '.D', enum_test, dememorization, batches, iterations)