import sys
import os
from .gui import *
from .polyviewer import PolyhedronViewer
from .horoviewer import HoroballViewer
from .CyOpenGL import GetColor
from .app_menus import browser_menus
from . import app_menus
from .number import Number
from . import database
from .exceptions import SnapPeaFatalError
from plink import LinkViewer, LinkEditor
from plink.ipython_tools import IPythonTkRoot
from spherogram.links.orthogonal import OrthogonalLinkDiagram
def clear_invariants(self):
    self.volume.set('')
    self.cs.set('')
    self.homology.set('')
    self.pi_one.set('')
    self.geodesics.delete(*self.geodesics.get_children())
    self.symmetry.set('')
    self.recompute_invariants = True