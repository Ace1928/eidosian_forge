from __future__ import annotations
import itertools
import math
import os
import subprocess
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import PeriodicSite, Species, Structure
from pymatgen.util.coord import in_coord_list
def display_warning(self, warning):
    """
        Args:
            warning (str): Warning.
        """
    self.warning_txt_mapper = vtk.vtkTextMapper()
    tprops = self.warning_txt_mapper.GetTextProperty()
    tprops.SetFontSize(14)
    tprops.SetFontFamilyToTimes()
    tprops.SetColor(1, 0, 0)
    tprops.BoldOn()
    tprops.SetJustificationToRight()
    self.warning_txt = f'WARNING : {warning}'
    self.warning_txt_actor = vtk.vtkActor2D()
    self.warning_txt_actor.VisibilityOn()
    self.warning_txt_actor.SetMapper(self.warning_txt_mapper)
    self.ren.AddActor(self.warning_txt_actor)
    self.warning_txt_mapper.SetInput(self.warning_txt)
    winsize = self.ren_win.GetSize()
    self.warning_txt_actor.SetPosition(winsize[0] - 10, 10)
    self.warning_txt_actor.VisibilityOn()