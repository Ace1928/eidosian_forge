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
def peek_nonwhitespace(self):
    """Return the first character from the buffer, do not include spaces."""
    b = ''.join(self.buffer).strip()
    if b:
        return b[0]
    else:
        return None