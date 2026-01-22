from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def set_original_taxon_order(self, value):
    """Included for backwards compatibility (DEPRECATED)."""
    warnings.warn('The set_original_taxon_order method has been deprecated and will likely be removed from Biopython in the near future. Please use the taxlabels attribute instead.', BiopythonDeprecationWarning)
    self.taxlabels = value