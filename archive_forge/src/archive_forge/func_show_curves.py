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
def show_curves(self):
    self.curves.delete(*self.curves.get_children())
    for curve in self.manifold.dual_curves(max_segments=self.max_segments):
        n = curve['index']
        parity = '+' if curve['parity'] == 1 else '-'
        length = Number(curve['filled_length'], precision=25)
        self.curves.insert('', 'end', values=(n, parity, length))