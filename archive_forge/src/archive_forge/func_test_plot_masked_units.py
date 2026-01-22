from datetime import datetime, timezone, timedelta
import platform
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.units as munits
from matplotlib.category import UnitData
import numpy as np
import pytest
@image_comparison(['plot_masked_units.png'], remove_text=True, style='mpl20', tol=0 if platform.machine() == 'x86_64' else 0.01)
def test_plot_masked_units():
    data = np.linspace(-5, 5)
    data_masked = np.ma.array(data, mask=(data > -2) & (data < 2))
    data_masked_units = Quantity(data_masked, 'meters')
    fig, ax = plt.subplots()
    ax.plot(data_masked_units)