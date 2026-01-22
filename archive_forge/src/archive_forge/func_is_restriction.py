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
def is_restriction(self, y):
    """Return if enzyme (name) is a known enzyme.

        True if y or eval(y) is a RestrictionType.
        """
    return isinstance(y, RestrictionType) or isinstance(eval(str(y)), RestrictionType)