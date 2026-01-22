from numpy.testing import assert_equal
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
import pandas
from statsmodels.regression.linear_model import OLS
def test_ltx_special_chars(self):
    desired = '\n\\begin{tabular}{lcc}\n\\toprule\n                    & \\textbf{\\#header1\\$} & \\textbf{header\\&$|$}  \\\\\n\\midrule\n\\textbf{$>$stub1\\%} &          22          &         1053          \\\\\n\\textbf{stub\\_2}    &          23          &        6250.4         \\\\\n\\bottomrule\n\\end{tabular}\n'
    actual = '\n%s\n' % tbl_c.as_latex_tabular(center=False)
    assert_equal(actual, desired)