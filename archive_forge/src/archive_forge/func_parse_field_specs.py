import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def parse_field_specs(field_specs):
    fields = []
    hier = []
    scent = []
    for fs in field_specs:
        fhs = fs.split(':')
        if len(fhs) == 3:
            scent.append(int(fhs[2]))
            hier.append(int(fhs[1]))
            fields.append(fhs[0])
        elif len(fhs) == 2:
            scent.append(-1)
            hier.append(int(fhs[1]))
            fields.append(fhs[0])
        elif len(fhs) == 1:
            scent.append(-1)
            hier.append(-1)
            fields.append(fhs[0])
    mxm = max(hier)
    for c in range(len(hier)):
        if hier[c] < 0:
            mxm += 1
            hier[c] = mxm
    hier = np.argsort(hier)[::-1]
    return (fields, hier, np.array(scent))