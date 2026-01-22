from pyomo.environ import (
import pyomo.common.unittest as unittest
import pyomo.contrib.viewer.qt as myqt
import pyomo.contrib.viewer.pyomo_viewer as pv
from pyomo.contrib.viewer.qt import available
@unittest.skipIf(not available, 'Qt packages are not available.')
def test_tree_expand_collapse(qtbot):
    m = get_model()
    mw, m = get_mainwindow(model=m, testing=True)
    mw.variables.treeView.expandAll()
    mw.variables.treeView.collapseAll()