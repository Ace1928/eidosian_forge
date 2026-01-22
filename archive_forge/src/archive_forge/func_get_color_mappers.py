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
def get_color_mappers(self, infer=False):
    if not infer:
        cmaps = []
        for view_prop in self.get_renderer().GetViewProps():
            if view_prop.IsA('vtkScalarBarActor'):
                name = view_prop.GetTitle()
                lut = view_prop.GetLookupTable()
                cmaps.append(self._vtklut2bkcmap(lut, name))
    else:
        infered_cmaps = {}
        for actor in self.get_renderer().GetActors():
            mapper = actor.GetMapper()
            cmap_name = mapper.GetArrayName()
            if cmap_name and cmap_name not in infered_cmaps:
                lut = mapper.GetLookupTable()
                infered_cmaps[cmap_name] = self._vtklut2bkcmap(lut, cmap_name)
        cmaps = infered_cmaps.values()
    return cmaps