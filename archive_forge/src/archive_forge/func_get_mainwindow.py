import os
import logging
import pyomo.contrib.viewer.report as rpt
import pyomo.environ as pyo
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.model_browser import ModelBrowser
from pyomo.contrib.viewer.residual_table import ResidualTable
from pyomo.contrib.viewer.model_select import ModelSelect
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.fileutils import this_file_dir
def get_mainwindow(model=None, show=True, ask_close=True, testing=False):
    """
    Create a UI MainWindow.

    Args:
        model: A Pyomo model to work with
        show: show the window after it is created
        ask_close: confirm close window
        testing: if True, expect testing
    Returns:
        (ui, model): ui is the MainWindow widget, and model is the linked Pyomo
            model.  If no model is provided a new ConcreteModel is created
    """
    if model is None:
        model = pyo.ConcreteModel(name='Default')
    ui = MainWindow(model=model, ask_close=ask_close, testing=testing)
    try:
        get_ipython().events.register('post_execute', ui.refresh_on_execute)
    except AttributeError:
        pass
    if show:
        ui.show()
    return (ui, model)