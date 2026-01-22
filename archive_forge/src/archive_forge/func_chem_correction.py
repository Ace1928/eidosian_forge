import math
import warnings
from Bio import SeqUtils, Seq
from Bio import BiopythonWarning
def chem_correction(melting_temp, DMSO=0, fmd=0, DMSOfactor=0.75, fmdfactor=0.65, fmdmethod=1, GC=None):
    """Correct a given Tm for DMSO and formamide.

    Please note that these corrections are +/- rough approximations.

    Arguments:
     - melting_temp: Melting temperature.
     - DMSO: Percent DMSO.
     - fmd: Formamide concentration in %(fmdmethod=1) or molar (fmdmethod=2).
     - DMSOfactor: How much should Tm decreases per percent DMSO. Default=0.65
       (von Ahsen et al. 2001). Other published values are 0.5, 0.6 and 0.675.
     - fmdfactor: How much should Tm decrease per percent formamide.
       Default=0.65. Several papers report factors between 0.6 and 0.72.
     - fmdmethod:

         1. Tm = Tm - factor(%formamide) (Default)
         2. Tm = Tm + (0.453(f(GC)) - 2.88) x [formamide]

       Here f(GC) is fraction of GC.
       Note (again) that in fmdmethod=1 formamide concentration is given in %,
       while in fmdmethod=2 it is given in molar.
     - GC: GC content in percent.

    Examples:
        >>> from Bio.SeqUtils import MeltingTemp as mt
        >>> mt.chem_correction(70)
        70
        >>> print('%0.2f' % mt.chem_correction(70, DMSO=3))
        67.75
        >>> print('%0.2f' % mt.chem_correction(70, fmd=5))
        66.75
        >>> print('%0.2f' % mt.chem_correction(70, fmdmethod=2, fmd=1.25,
        ...                                    GC=50))
        66.68

    """
    if DMSO:
        melting_temp -= DMSOfactor * DMSO
    if fmd:
        if fmdmethod == 1:
            melting_temp -= fmdfactor * fmd
        if fmdmethod == 2:
            if GC is None or GC < 0:
                raise ValueError("'GC' is missing or negative")
            melting_temp += (0.453 * (GC / 100.0) - 2.88) * fmd
        if fmdmethod not in (1, 2):
            raise ValueError("'fmdmethod' must be 1 or 2")
    return melting_temp