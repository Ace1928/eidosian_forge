import base64
import os
import sys
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pytest
from bokeh.models import ColorBar
from panel.models.vtk import (
from panel.pane import VTK, PaneBase, VTKVolume
from panel.pane.vtk.vtk import (
def make_image_data():
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(3, 4, 5)
    image_data.AllocateScalars(vtk.VTK_DOUBLE, 1)
    dims = image_data.GetDimensions()
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                image_data.SetScalarComponentFromDouble(x, y, z, 0, np.random.rand())
    return image_data