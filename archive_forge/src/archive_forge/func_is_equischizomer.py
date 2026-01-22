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
def is_equischizomer(cls, other):
    """Test for real isoschizomer.

        True if other is an isoschizomer of RE, but not an neoschizomer,
        else False.

        Equischizomer: same site, same position of restriction.

        >>> from Bio.Restriction import SacI, SstI, SmaI, XmaI
        >>> SacI.is_equischizomer(SstI)
        True
        >>> SmaI.is_equischizomer(XmaI)
        False

        """
    return not cls != other