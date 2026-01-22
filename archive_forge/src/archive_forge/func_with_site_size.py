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
def with_site_size(self, site_size, dct=None):
    """Return only results form enzymes with a given site size."""
    sites = [name for name in self if name.size == site_size]
    if not dct:
        return RestrictionBatch(sites).search(self.sequence)
    return {k: v for k, v in dct.items() if k in site_size}