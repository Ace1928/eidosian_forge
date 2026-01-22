import numpy as np
from numpy.testing import assert_equal
from statsmodels.iolib.table import Cell, SimpleTable
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
def test_customlabel(self):
    tbl = SimpleTable(table1data, test1header, test1stubs, txt_fmt=txt_fmt1)
    tbl[1][1].data = np.nan
    tbl.label_cells(custom_labeller)
    desired = '\n*****************************\n*       * header1 * header2 *\n*****************************\n* stub1 *    --   *       1 *\n* stub2 *    2.00 *       3 *\n*****************************\n'
    actual = '\n%s\n' % tbl.as_text(missing='--')
    assert_equal(actual, desired)