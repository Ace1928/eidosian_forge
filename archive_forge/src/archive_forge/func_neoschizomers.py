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
def neoschizomers(cls, batch=None):
    """List neoschizomers of the enzyme.

        Return a tuple of all the neoschizomers of RE.
        If batch is supplied it is used instead of the default AllEnzymes.

        Neoschizomer: same site, different position of restriction.
        """
    if not batch:
        batch = AllEnzymes
    r = sorted((x for x in batch if cls >> x))
    return r