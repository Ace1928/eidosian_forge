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
def validate_segments(self, P):
    try:
        new_max = int(P)
        if self.max_segments != new_max:
            self.max_segments = new_max
            self.segment_var.set(str(self.max_segments))
            self.show_curves()
    except ValueError:
        self.root.after_idle(self.segment_var.set, str(self.max_segments))
        return False
    return True