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
def current_suppliers(self):
    """List the current suppliers for the restriction batch.

        Return a sorted list of the suppliers which have been used to
        create the batch.
        """
    suppl_list = sorted((suppliers_dict[x][0] for x in self.suppliers))
    return suppl_list