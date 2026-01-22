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
def validate_cutoff(self, P):
    try:
        cutoff = float(P)
        if self.length_cutoff != cutoff:
            self.length_cutoff = cutoff
            self.update_length_spectrum()
    except ValueError:
        self.after_idle(self.cutoff_var.set, str(self.length_cutoff))
        return False
    return True