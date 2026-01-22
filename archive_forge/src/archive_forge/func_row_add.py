from copy import copy
from ase.gui.i18n import _
import numpy as np
import ase
import ase.data
import ase.gui.ui as ui
from ase.cluster.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic
from ase.cluster.hexagonal import HexagonalClosedPacked, Graphite
from ase.cluster import wulff_construction
from ase.gui.widgets import Element, pybutton
import ase
import ase
from ase.cluster import wulff_construction
def row_add(self, widget=None):
    """Add a row to the list of directions."""
    if self.fourindex:
        n = 4
    else:
        n = 3
    idx = tuple((a.value for a in self.new_direction[1:1 + 2 * n:2]))
    if not any(idx):
        ui.error(_('At least one index must be non-zero'), '')
        return
    if n == 4 and sum(idx) != 0:
        ui.error(_('Invalid hexagonal indices', 'The sum of the first three numbers must be zero'))
        return
    new = [idx, 5, 1.0]
    if self.method.value == 'wulff':
        new[1] = self.new_direction[-2].value
    else:
        new[2] = self.new_direction[-2].value
    self.direction_table.append(new)
    self.add_direction(*new)
    self.update()