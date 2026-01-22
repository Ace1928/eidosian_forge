import numpy as np
from ase.data import atomic_numbers
from ase.ga.offspring_creator import OffspringCreator
def get_row_column(element):
    """Returns the row and column of the element in the periodic table.
    Note that Lanthanides and Actinides are defined to be group (column)
    3 elements"""
    t = mendeleiev_table()
    en = (element, atomic_numbers[element])
    for i in range(len(t)):
        for j in range(len(t[i])):
            if en == t[i][j]:
                return (i, j)
            elif isinstance(t[i][j], list):
                if en in t[i][j]:
                    return (i, 3)