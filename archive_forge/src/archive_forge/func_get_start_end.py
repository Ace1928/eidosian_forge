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
def get_start_end(sequence, skiplist=('-', '?')):
    """Return position of first and last character which is not in skiplist.

    Skiplist defaults to ['-','?'].
    """
    length = len(sequence)
    if length == 0:
        return (None, None)
    end = length - 1
    while end >= 0 and sequence[end] in skiplist:
        end -= 1
    start = 0
    while start < length and sequence[start] in skiplist:
        start += 1
    if start == length and end == -1:
        return (-1, -1)
    else:
        return (start, end)