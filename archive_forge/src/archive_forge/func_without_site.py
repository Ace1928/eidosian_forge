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
def without_site(self, dct=None):
    """Return only results from enzymes that don't cut the sequence."""
    if not dct:
        dct = self.mapping
    return {k: v for k, v in dct.items() if not v}