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
def set_structure_data(self, *args):
    """Called when the user presses [Get structure]."""
    z = self.element.Z
    if z is None:
        return
    ref = ase.data.reference_states[z]
    if ref is None:
        structure = None
    else:
        structure = ref['symmetry']
    if ref is None or structure not in [s[0] for s in self.structure_data]:
        ui.error(_('Unsupported or unknown structure'), _('Element = {0}, structure = {1}').format(self.element.symbol, structure))
        return
    self.structure.value = structure
    a = ref['a']
    self.a.value = a
    self.fourindex = self.needs_4index[structure]
    if self.fourindex:
        try:
            c = ref['c']
        except KeyError:
            c = ref['c/a'] * a
        self.c.value = c