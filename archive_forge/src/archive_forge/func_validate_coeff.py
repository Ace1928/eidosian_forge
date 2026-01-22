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
def validate_coeff(self, P, W):
    tkname, cusp, curve = W.split(':')
    cusp, curve = (int(cusp), int(curve))
    try:
        float(P)
    except ValueError:
        var = self.filling_vars[cusp][curve]
        if P == '':
            var.set('0')
        else:
            value = '%g' % self.manifold.cusp_info()[cusp].filling[curve]
            value = '0' if value == '-0' else value
            var.set(value)
        return False
    return True