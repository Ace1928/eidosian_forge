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
def makeinfo(self):
    """Fill in information field about the atoms.

        Also turns the Wulff construction buttons [Larger] and
        [Smaller] on and off.
        """
    if self.atoms is None:
        self.info[1].text = '-'
        self.info[3].text = '-'
    else:
        at_vol = self.get_atomic_volume()
        dia = 2 * (3 * len(self.atoms) * at_vol / (4 * np.pi)) ** (1 / 3)
        self.info[1].text = str(len(self.atoms))
        self.info[3].text = u'{0:.1f} Å'.format(dia)
    if self.method.value == 'wulff':
        if self.smaller_button is not None:
            self.smaller_button.active = self.atoms is not None
            self.larger_button.active = self.atoms is not None