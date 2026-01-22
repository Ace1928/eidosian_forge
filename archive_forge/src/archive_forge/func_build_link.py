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
def build_link(self):
    if self.manifold.DT_code() is not None:
        data = None
        try:
            data = OrthogonalLinkDiagram(self.manifold.link()).plink_data()
        except:
            if self.manifold.LE:
                data = self.manifold.LE.pickle()
        if data:
            return LinkTab(data, self)