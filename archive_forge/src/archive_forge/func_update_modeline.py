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
def update_modeline(self):
    modeline = self.modeline
    modeline.config(state=Tk_.NORMAL)
    modeline.delete(1.0, Tk_.END)
    if not self.manifold.is_orientable():
        modeline.insert(Tk_.END, 'Non-orientable; ')
    modeline.insert(Tk_.END, '%s tetrahedra; %s' % (self.manifold.num_tetrahedra(), self.manifold.solution_type()))
    if len(self.dirichlet) == 0:
        modeline.insert(Tk_.END, '  * failed to compute Dirichlet domain!', 'alert')
    modeline.config(state=Tk_.DISABLED)