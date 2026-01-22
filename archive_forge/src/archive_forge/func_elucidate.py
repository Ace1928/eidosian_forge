import warnings
import re
import string
import itertools
from Bio.Seq import Seq, MutableSeq
from Bio.Restriction.Restriction_Dictionary import rest_dict as enzymedict
from Bio.Restriction.Restriction_Dictionary import typedict
from Bio.Restriction.Restriction_Dictionary import suppliers as suppliers_dict
from Bio.Restriction.PrintFormat import PrintFormat
from Bio import BiopythonWarning
@classmethod
def elucidate(cls):
    """Return a string representing the recognition site and cuttings.

        Return a representation of the site with the cut on the (+) strand
        represented as '^' and the cut on the (-) strand as '_'.
        ie:

        >>> from Bio.Restriction import EcoRI, KpnI, EcoRV, SnaI
        >>> EcoRI.elucidate()   # 5' overhang
        'G^AATT_C'
        >>> KpnI.elucidate()    # 3' overhang
        'G_GTAC^C'
        >>> EcoRV.elucidate()   # blunt
        'GAT^_ATC'
        >>> SnaI.elucidate()     # NotDefined, cut profile unknown.
        '? GTATAC ?'
        >>>

        """
    return f'? {cls.site} ?'