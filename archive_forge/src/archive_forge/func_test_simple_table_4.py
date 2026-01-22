from numpy.testing import assert_equal
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
import pandas
from statsmodels.regression.linear_model import OLS
def test_simple_table_4(self):
    txt_fmt1 = dict(data_fmts=['%3.2f', '%d'], empty_cell=' ', colwidths=1, colsep=' * ', row_pre='* ', row_post=' *', table_dec_above='*', table_dec_below='*', header_dec_below='*', header_fmt='%s', stub_fmt='%s', title_align='r', header_align='r', data_aligns='r', stubs_align='l', fmt='txt')
    ltx_fmt1 = default_latex_fmt.copy()
    html_fmt1 = default_html_fmt.copy()
    cell0data = 0.0
    cell1data = 1
    row0data = [cell0data, cell1data]
    row1data = [2, 3.333]
    table1data = [row0data, row1data]
    test1stubs = ('stub1', 'stub2')
    test1header = ('header1', 'header2')
    tbl = SimpleTable(table1data, test1header, test1stubs, txt_fmt=txt_fmt1, ltx_fmt=ltx_fmt1, html_fmt=html_fmt1)

    def test_txt_fmt1(self):
        desired = '\n*****************************\n*       * header1 * header2 *\n*****************************\n* stub1 *    0.00 *       1 *\n* stub2 *    2.00 *       3 *\n*****************************\n'
        actual = '\n%s\n' % tbl.as_text()
        assert_equal(actual, desired)

    def test_ltx_fmt1(self):
        desired = '\n\\begin{tabular}{lcc}\n\\toprule\n               & \\textbf{header1} & \\textbf{header2}  \\\\\n\\midrule\n\\textbf{stub1} &       0.0        &        1          \\\\\n\\textbf{stub2} &        2         &      3.333        \\\\\n\\bottomrule\n\\end{tabular}\n'
        actual = '\n%s\n' % tbl.as_latex_tabular(center=False)
        assert_equal(actual, desired)
        desired_centered = '\n\\begin{center}\n%s\n\\end{center}\n' % desired[1:-1]
        actual_centered = '\n%s\n' % tbl.as_latex_tabular()
        assert_equal(actual_centered, desired_centered)

    def test_html_fmt1(self):
        desired = '\n<table class="simpletable">\n<tr>\n    <td></td>    <th>header1</th> <th>header2</th>\n</tr>\n<tr>\n  <th>stub1</th>   <td>0.0</td>      <td>1</td>   \n</tr>\n<tr>\n  <th>stub2</th>    <td>2</td>     <td>3.333</td> \n</tr>\n</table>\n'
        actual = '\n%s\n' % tbl.as_html()
        assert_equal(actual, desired)
    test_txt_fmt1(self)
    test_ltx_fmt1(self)
    test_html_fmt1(self)