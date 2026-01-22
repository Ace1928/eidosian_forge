import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataModel
import pyomo.contrib.viewer.qt as myqt
from pyomo.common.dependencies import DeferredImportError
def test_create_tree_var(self):
    ui_data = UIData(model=self.m)
    data_model = ComponentDataModel(parent=None, ui_data=ui_data)
    assert len(data_model.rootItems) == 1
    assert data_model.rootItems[0].data == self.m
    children = data_model.rootItems[0].children
    assert children[0].data == self.m.y
    assert children[1].data == self.m.z
    assert children[2].data == self.m.x
    assert children[3].data == self.m.b1
    root_index = data_model.index(0, 0)
    assert data_model.data(root_index) == 'tm'
    zidx = data_model.index(0, 0, parent=root_index)
    assert data_model.data(zidx) == 'y'
    zidx = data_model.index(1, 0, parent=root_index)
    assert data_model.data(zidx) == 'z'
    xidx = data_model.index(2, 0, parent=root_index)
    assert data_model.data(xidx) == 'x'
    b1idx = data_model.index(3, 0, parent=root_index)
    assert data_model.data(b1idx) == 'b1'
    idx = data_model.index(0, 0, parent=zidx)
    assert data_model.data(idx) == 'z[0]'
    idx = data_model.index(0, 1, parent=zidx)
    assert abs(data_model.data(idx) - 2.0) < 0.0001
    idx = data_model.index(1, 0, parent=zidx)
    assert data_model.data(idx) == 'z[1]'
    idx = data_model.index(1, 1, parent=zidx)
    assert abs(data_model.data(idx) - 2.0) < 0.0001
    assert 'test doc' == data_model.data(zidx, role=myqt.Qt.ItemDataRole.ToolTipRole)
    assert myqt.QtCore.Qt.blue == data_model.data(zidx, role=myqt.Qt.ItemDataRole.ForegroundRole)
    assert myqt.QtCore.Qt.black == data_model.data(root_index, role=myqt.Qt.ItemDataRole.ForegroundRole)