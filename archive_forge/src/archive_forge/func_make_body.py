import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def make_body(self, atoms1, atoms2, csv=False):
    field_data = np.array([get_field_data(atoms1, atoms2, field) for field in self.fields])
    sorting_array = field_data * self.scent[:, np.newaxis]
    sorting_array = sorting_array[self.hier]
    sorting_array = prec_round(sorting_array, self.tableformat.precision)
    field_data = field_data[:, np.lexsort(sorting_array)].transpose()
    if csv:
        rowformat = ','.join(['{:h}' if field == 'el' else '{{:.{}E}}'.format(self.tableformat.precision) for field in self.fields])
    else:
        rowformat = ''.join([self.tableformat.fmt[field] for field in self.fields])
    body = [self.tableformat.formatter(rowformat, *row) for row in field_data]
    return '\n'.join(body)