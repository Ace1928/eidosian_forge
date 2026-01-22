from pyomo.environ import (
import pyomo.common.unittest as unittest
import pyomo.contrib.viewer.qt as myqt
import pyomo.contrib.viewer.pyomo_viewer as pv
from pyomo.contrib.viewer.qt import available
@unittest.skipIf(not available, 'Qt packages are not available.')
def test_model_information(qtbot):
    m = get_model()
    mw, m = get_mainwindow(model=m, testing=True)
    mw.model_information()
    assert isinstance(mw._dialog, QMessageBox)
    text = mw._dialog.text()
    mw._dialog.close()
    text = text.split('\n')
    assert str(text[0]).startswith('8')
    assert str(text[1]).startswith('7')
    assert str(text[2]).startswith('7')
    assert str(text[3]).startswith('0')
    assert hasattr(mw, 'menuBar')
    assert isinstance(mw.variables, ModelBrowser)
    assert isinstance(mw.constraints, ModelBrowser)
    assert isinstance(mw.expressions, ModelBrowser)
    assert isinstance(mw.parameters, ModelBrowser)