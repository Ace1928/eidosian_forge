import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def field_specs_on_conditions(calculator_outputs, rank_order):
    if calculator_outputs:
        field_specs = ['i:0', 'el', 'd', 'rd', 'df', 'rdf']
    else:
        field_specs = ['i:0', 'el', 'dx', 'dy', 'dz', 'd', 'rd']
    if rank_order is not None:
        field_specs[0] = 'i:1'
        if rank_order in field_specs:
            for c, i in enumerate(field_specs):
                if i == rank_order:
                    field_specs[c] = i + ':0:1'
        else:
            field_specs.append(rank_order + ':0:1')
    else:
        field_specs[0] = field_specs[0] + ':1'
    return field_specs