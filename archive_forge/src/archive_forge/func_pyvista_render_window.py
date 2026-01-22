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
@pytest.fixture
def pyvista_render_window():
    """
    Allow to download and create a more complex example easily
    """
    from pyvista import examples
    sphere = pv.Sphere()
    globe = examples.load_globe()
    head = examples.download_head()
    uniform = examples.load_uniform()
    scalars = sphere.points[:, 2]
    sphere.point_data['values'] = scalars
    sphere.set_active_scalars('values')
    uniform.set_active_scalars('Spatial Cell Data')
    threshed = uniform.threshold_percent([0.15, 0.5], invert=True)
    bodies = threshed.split_bodies()
    mapper = vtk.vtkCompositePolyDataMapper2()
    mapper.SetInputDataObject(0, bodies)
    multiblock = vtk.vtkActor()
    multiblock.SetMapper(mapper)
    pl = pv.Plotter()
    pl.add_mesh(globe)
    pl.add_mesh(sphere)
    pl.add_mesh(uniform)
    pl.add_actor(multiblock)
    pl.add_volume(head)
    yield pl.ren_win
    pv.close_all()