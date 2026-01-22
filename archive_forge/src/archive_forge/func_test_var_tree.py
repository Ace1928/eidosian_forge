from pyomo.environ import (
import pyomo.common.unittest as unittest
import pyomo.contrib.viewer.qt as myqt
import pyomo.contrib.viewer.pyomo_viewer as pv
from pyomo.contrib.viewer.qt import available
@unittest.skipIf(not available, 'Qt packages are not available.')
def test_var_tree(qtbot):
    m = get_model()
    mw, m = get_mainwindow(model=m, testing=True)
    qtbot.addWidget(mw)
    mw.variables.treeView.expandAll()
    root_index = mw.variables.datmodel.index(0, 0)
    z_index = mw.variables.datmodel.index(1, 0, parent=root_index)
    z_val_index = mw.variables.datmodel.index(1, 1, parent=root_index)
    z1_val_index = mw.variables.datmodel.index(0, 1, parent=z_index)
    assert mw.variables.datmodel.data(z1_val_index) == 2.0
    mw.variables.treeView.setCurrentIndex(z1_val_index)
    mw.variables.treeView.openPersistentEditor(z1_val_index)
    d = mw.variables.treeView.itemDelegate()
    w = mw.variables.treeView.indexWidget(z1_val_index)
    w.setText('Not a number')
    d.setModelData(w, mw.variables.datmodel, z1_val_index)
    assert value(m.z[0]) - 2.0 < 1e-06
    w.setText('1e5')
    d.setModelData(w, mw.variables.datmodel, z1_val_index)
    assert value(m.z[0]) - 100000.0 < 1e-06
    w.setText('false')
    d.setModelData(w, mw.variables.datmodel, z1_val_index)
    assert value(m.z[0]) - 0 < 1e-06
    w.setText('true')
    d.setModelData(w, mw.variables.datmodel, z1_val_index)
    assert value(m.z[0]) - 1 < 1e-06
    mw.variables.treeView.closePersistentEditor(z1_val_index)
    w.setText('2')
    d.setModelData(w, mw.variables.datmodel, z1_val_index)
    assert value(m.z[0]) - 2 < 1e-06
    mw.variables.treeView.closePersistentEditor(z1_val_index)