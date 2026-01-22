from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixedPlane
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
import re
import pytest
from ase.cli.template import prec_round, slice_split, \
from ase.io import read
def test_template_classes(traj):
    prec = 4
    tableformat = TableFormat(precision=prec, representation='f', midrule='|')
    table = Table(field_specs=('dx', 'dy', 'dz'), tableformat=tableformat)
    images = read(traj, ':')
    table_out = table.make(images[0], images[1]).split('\n')
    for counter, row in enumerate(table_out):
        if '|' in row:
            break
    row = table_out[counter + 2]
    assert 'E' not in table_out[counter + 2]
    row = re.sub('\\s+', ',', table_out[counter + 2]).split(',')[1:-1]
    assert len(row[0]) >= prec