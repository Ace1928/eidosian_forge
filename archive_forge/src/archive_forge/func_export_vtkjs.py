from __future__ import annotations
import base64
import json
import sys
import zipfile
from abc import abstractmethod
from typing import (
from urllib.request import urlopen
import numpy as np
import param
from bokeh.models import LinearColorMapper
from bokeh.util.serialization import make_globally_unique_id
from pyviz_comms import JupyterComm
from ...param import ParamMethod
from ...util import isfile, lazy_load
from ..base import PaneBase
from ..plot import Bokeh
from .enums import PRESET_CMAPS
def export_vtkjs(self, filename: str | IO='vtk_panel.vtkjs'):
    """
        Exports current VTK data to .vtkjs file.

        Arguments
        ---------
        filename: str | IO
        """
    _, vtkjs = self._get_vtkjs()
    if hasattr(filename, 'write'):
        filename.write(vtkjs)
    else:
        with open(filename, 'wb') as f:
            f.write(vtkjs)