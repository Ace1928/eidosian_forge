from numpy.testing import assert_equal
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
import pandas
from statsmodels.regression.linear_model import OLS
def test_default_alignment(self):
    desired = '\n=====================\n      header1 header2\n---------------------\nstub1 1.30312    2.73\nstub2 1.95038     2.6\n---------------------\n'
    test1data = [[1.30312, 2.73], [1.95038, 2.6]]
    test1stubs = ('stub1', 'stub2')
    test1header = ('header1', 'header2')
    actual = SimpleTable(test1data, test1header, test1stubs, txt_fmt=default_txt_fmt)
    actual = '\n%s\n' % actual.as_text()
    assert_equal(desired, str(actual))