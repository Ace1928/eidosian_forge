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
def update_size_natoms(self, widget=None):
    at_vol = self.get_atomic_volume()
    dia = 2.0 * (3 * self.size_natoms.value * at_vol / (4 * np.pi)) ** (1 / 3)
    self.size_diameter.value = dia
    self.update()