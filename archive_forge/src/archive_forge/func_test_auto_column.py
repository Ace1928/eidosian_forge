import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.table import CustomCell, Table
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
@image_comparison(['table_auto_column.png'])
def test_auto_column():
    fig = plt.figure()
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.axis('off')
    tb1 = ax1.table(cellText=[['Fit Text', 2], ['very long long text, Longer text than default', 1]], rowLabels=['A', 'B'], colLabels=['Col1', 'Col2'], loc='center')
    tb1.auto_set_font_size(False)
    tb1.set_fontsize(12)
    tb1.auto_set_column_width([-1, 0, 1])
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.axis('off')
    tb2 = ax2.table(cellText=[['Fit Text', 2], ['very long long text, Longer text than default', 1]], rowLabels=['A', 'B'], colLabels=['Col1', 'Col2'], loc='center')
    tb2.auto_set_font_size(False)
    tb2.set_fontsize(12)
    tb2.auto_set_column_width((-1, 0, 1))
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.axis('off')
    tb3 = ax3.table(cellText=[['Fit Text', 2], ['very long long text, Longer text than default', 1]], rowLabels=['A', 'B'], colLabels=['Col1', 'Col2'], loc='center')
    tb3.auto_set_font_size(False)
    tb3.set_fontsize(12)
    tb3.auto_set_column_width(-1)
    tb3.auto_set_column_width(0)
    tb3.auto_set_column_width(1)
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.axis('off')
    tb4 = ax4.table(cellText=[['Fit Text', 2], ['very long long text, Longer text than default', 1]], rowLabels=['A', 'B'], colLabels=['Col1', 'Col2'], loc='center')
    tb4.auto_set_font_size(False)
    tb4.set_fontsize(12)
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match="'col' must be an int or sequence of ints"):
        tb4.auto_set_column_width('-101')
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match="'col' must be an int or sequence of ints"):
        tb4.auto_set_column_width(['-101'])