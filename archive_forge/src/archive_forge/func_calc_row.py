import math
import sys
from Bio import MissingPythonDependencyError
def calc_row(clade):
    for subclade in clade:
        if subclade not in heights:
            calc_row(subclade)
    heights[clade] = (heights[clade.clades[0]] + heights[clade.clades[-1]]) / 2.0