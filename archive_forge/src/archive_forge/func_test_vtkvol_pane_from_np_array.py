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
@vtk_available
def test_vtkvol_pane_from_np_array(document, comm):
    pane = VTKVolume()
    model = pane.get_root(document, comm=comm)
    pane.object = np.ones((10, 10, 10))
    from operator import eq
    assert isinstance(model, VTKVolumePlot)
    assert pane._models[model.ref['id']][0] is model
    assert np.all(np.frombuffer(base64.b64decode(model.data['buffer'].encode())) == 1)
    assert all([eq(getattr(pane, k), getattr(model, k)) for k in ['slice_i', 'slice_j', 'slice_k']])
    pane.object = 2 * np.ones((10, 10, 10))
    assert np.all(np.frombuffer(base64.b64decode(model.data['buffer'].encode())) == 2)
    pane.max_data_size = 0.1
    data = (255 * np.random.rand(50, 50, 50)).astype(np.uint8)
    assert data.nbytes / 1000000.0 > 0.1
    pane.object = data
    data_model = np.frombuffer(base64.b64decode(model.data['buffer'].encode()))
    assert data_model.nbytes / 1000000.0 <= 0.1
    data = np.random.rand(50, 50, 50)
    assert data.nbytes / 1000000.0 > 0.1
    pane.object = data
    data_model = np.frombuffer(base64.b64decode(model.data['buffer'].encode()), dtype=np.float64)
    assert data_model.nbytes / 1000000.0 <= 0.1
    param = pane._process_property_change({'slice_i': (np.cbrt(data_model.size) - 1) // 2})
    assert param == {'slice_i': (50 - 1) // 2}
    pane._cleanup(model)
    assert pane._models == {}