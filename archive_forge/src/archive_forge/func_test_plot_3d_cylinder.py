from sympy.external.importtools import import_module
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
def test_plot_3d_cylinder():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(1 / y, [x, 0, 6.282, 4], [y, -1, 1, 4], 'mode=polar;style=solid', visible=False)
    p.wait_for_calculations()