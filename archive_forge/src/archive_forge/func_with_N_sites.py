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
def with_N_sites(self, N, dct=None):
    """Return only results from enzymes that cut the sequence N times."""
    if not dct:
        dct = self.mapping
    return {k: v for k, v in dct.items() if len(v) == N}