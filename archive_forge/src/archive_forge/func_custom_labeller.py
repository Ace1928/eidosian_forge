import numpy as np
from numpy.testing import assert_equal
from statsmodels.iolib.table import Cell, SimpleTable
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
def custom_labeller(cell):
    if cell.data is np.nan:
        return 'missing'