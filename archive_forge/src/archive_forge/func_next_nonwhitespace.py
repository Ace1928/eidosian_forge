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
def next_nonwhitespace(self):
    """Check for next non whitespace character in NEXUS file."""
    while True:
        p = next(self)
        if p is None:
            break
        if p not in WHITESPACE:
            return p
    return None