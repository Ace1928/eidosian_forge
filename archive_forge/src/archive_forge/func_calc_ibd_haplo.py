import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def calc_ibd_haplo(self, fname, stat='a', scale='Log', min_dist=1e-05):
    """Calculate isolation by distance statistics for haploid data.

        See _calc_ibd for parameter details.

        Note that each pop can only have a single individual and
        the individual name has to be the sample coordinates.
        """
    return self._calc_ibd(fname, 6, stat, scale, min_dist)