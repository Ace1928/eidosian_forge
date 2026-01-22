from .gui import *
from .CyOpenGL import (HoroballScene, OpenGLOrthoWidget,
from plink.ipython_tools import IPythonTkRoot
import os
import sys
def update_radius(self, event=None):
    index = self.moving_cusp
    value = self.cusp_sliders[index].get()
    if value == self.last_slider_value:
        return
    if self.busy_drawing:
        return
    self.last_slider_value = value
    stop = float(self.nbhd.stopping_displacement(index))
    disp = value * stop / 100.0
    self.nbhd.set_displacement(disp, index)
    self.busy_drawing = True
    self.rebuild(full_list=False)
    self.busy_drawing = False