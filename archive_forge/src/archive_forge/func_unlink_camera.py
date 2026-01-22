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
def unlink_camera(self):
    """
        Create a fresh vtkCamera instance and set it to the renderer
        """
    import vtk
    old_camera = self.vtk_camera
    new_camera = vtk.vtkCamera()
    self.vtk_camera = new_camera
    exclude_properties = ['mtime', 'projectionMatrix', 'viewMatrix', 'physicalTranslation', 'physicalScale', 'physicalViewUp', 'physicalViewNorth', 'remoteId']
    if self.camera is not None:
        for k, v in self.camera.items():
            if k not in exclude_properties:
                if isinstance(v, list):
                    getattr(new_camera, 'Set' + k[0].capitalize() + k[1:])(*v)
                else:
                    getattr(new_camera, 'Set' + k[0].capitalize() + k[1:])(v)
    else:
        new_camera.DeepCopy(old_camera)