from sympy.core.singleton import S
from sympy.printing.tableform import TableForm
from sympy.printing.latex import latex
from sympy.abc import x
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import raises
from textwrap import dedent
def test_TableForm_latex():
    s = latex(TableForm([[0, x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]], wipe_zeros=True, headings=('automatic', 'automatic')))
    assert s == '\\begin{tabular}{r l l}\n & 1 & 2 \\\\\n\\hline\n1 &   & $x^{3}$ \\\\\n2 & $c$ & $\\frac{1}{4}$ \\\\\n3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'
    s = latex(TableForm([[0, x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]], wipe_zeros=True, headings=('automatic', 'automatic'), alignments='l'))
    assert s == '\\begin{tabular}{r l l}\n & 1 & 2 \\\\\n\\hline\n1 &   & $x^{3}$ \\\\\n2 & $c$ & $\\frac{1}{4}$ \\\\\n3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'
    s = latex(TableForm([[0, x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]], wipe_zeros=True, headings=('automatic', 'automatic'), alignments='l' * 3))
    assert s == '\\begin{tabular}{l l l}\n & 1 & 2 \\\\\n\\hline\n1 &   & $x^{3}$ \\\\\n2 & $c$ & $\\frac{1}{4}$ \\\\\n3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'
    s = latex(TableForm([['a', x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]], headings=('automatic', 'automatic')))
    assert s == '\\begin{tabular}{r l l}\n & 1 & 2 \\\\\n\\hline\n1 & $a$ & $x^{3}$ \\\\\n2 & $c$ & $\\frac{1}{4}$ \\\\\n3 & $\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'
    s = latex(TableForm([['a', x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]], formats=['(%s)', None], headings=('automatic', 'automatic')))
    assert s == '\\begin{tabular}{r l l}\n & 1 & 2 \\\\\n\\hline\n1 & (a) & $x^{3}$ \\\\\n2 & (c) & $\\frac{1}{4}$ \\\\\n3 & (sqrt(x)) & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'

    def neg_in_paren(x, i, j):
        if i % 2:
            return ('(%s)' if x < 0 else '%s') % x
        else:
            pass
    s = latex(TableForm([[-1, 2], [-3, 4]], formats=[neg_in_paren] * 2, headings=('automatic', 'automatic')))
    assert s == '\\begin{tabular}{r l l}\n & 1 & 2 \\\\\n\\hline\n1 & -1 & 2 \\\\\n2 & (-3) & 4 \\\\\n\\end{tabular}'
    s = latex(TableForm([['a', x ** 3], ['c', S.One / 4], [sqrt(x), sin(x ** 2)]]))
    assert s == '\\begin{tabular}{l l}\n$a$ & $x^{3}$ \\\\\n$c$ & $\\frac{1}{4}$ \\\\\n$\\sqrt{x}$ & $\\sin{\\left(x^{2} \\right)}$ \\\\\n\\end{tabular}'