from .gui import *
from .CyOpenGL import (HoroballScene, OpenGLOrthoWidget,
from plink.ipython_tools import IPythonTkRoot
import os
import sys
def set_cutoff(self, event=None):
    try:
        self.cutoff = float(self.cutoff_var.get())
        self.scene.set_cutoff(self.cutoff)
        self.rebuild()
    except Exception:
        pass
    self.cutoff_var.set('%.4f' % self.cutoff)