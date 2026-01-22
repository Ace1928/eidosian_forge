import glob
import re
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
@staticmethod
def parse_title(line):
    info = dict()
    tokens = line.split()
    num_tokens = len(tokens)
    if num_tokens <= 1:
        return info
    info['name'] = tokens[1]
    if num_tokens <= 2:
        return info
    info['pressure'] = float(tokens[2])
    if num_tokens <= 4:
        return info
    info['energy'] = float(tokens[4])
    idx = 7
    if tokens[idx][0] != '(':
        idx += 1
    if num_tokens <= idx:
        return info
    info['spacegroup'] = tokens[idx][1:len(tokens[idx]) - 1]
    if num_tokens <= idx + 3:
        return info
    info['times_found'] = int(tokens[idx + 3])
    return info