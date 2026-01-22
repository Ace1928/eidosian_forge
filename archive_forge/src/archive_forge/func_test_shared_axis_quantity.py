from datetime import datetime, timezone, timedelta
import platform
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.units as munits
from matplotlib.category import UnitData
import numpy as np
import pytest
def test_shared_axis_quantity(quantity_converter):
    munits.registry[Quantity] = quantity_converter
    x = Quantity(np.linspace(0, 1, 10), 'hours')
    y1 = Quantity(np.linspace(1, 2, 10), 'feet')
    y2 = Quantity(np.linspace(3, 4, 10), 'feet')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', sharey='all')
    ax1.plot(x, y1)
    ax2.plot(x, y2)
    assert ax1.xaxis.get_units() == ax2.xaxis.get_units() == 'hours'
    assert ax2.yaxis.get_units() == ax2.yaxis.get_units() == 'feet'
    ax1.xaxis.set_units('seconds')
    ax2.yaxis.set_units('inches')
    assert ax1.xaxis.get_units() == ax2.xaxis.get_units() == 'seconds'
    assert ax1.yaxis.get_units() == ax2.yaxis.get_units() == 'inches'