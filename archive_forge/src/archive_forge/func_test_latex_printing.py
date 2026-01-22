from sympy.core.function import Function
from sympy.integrals.integrals import Integral
from sympy.printing.latex import latex
from sympy.printing.pretty import pretty as xpretty
from sympy.vector import CoordSys3D, Del, Vector, express
from sympy.abc import a, b, c
from sympy.testing.pytest import XFAIL
def test_latex_printing():
    assert latex(v[0]) == '\\mathbf{\\hat{0}}'
    assert latex(v[1]) == '\\mathbf{\\hat{i}_{N}}'
    assert latex(v[2]) == '- \\mathbf{\\hat{i}_{N}}'
    assert latex(v[5]) == '\\left(a\\right)\\mathbf{\\hat{i}_{N}} + ' + '\\left(- b\\right)\\mathbf{\\hat{j}_{N}}'
    assert latex(v[6]) == '\\left(\\mathbf{{x}_{N}} + a^{2}\\right)\\mathbf{\\hat{i}_' + '{N}} + \\mathbf{\\hat{k}_{N}}'
    assert latex(v[8]) == '\\mathbf{\\hat{j}_{N}} + \\left(\\mathbf{{x}_' + '{C}}^{2} - \\int f{\\left(b \\right)}\\,' + ' db\\right)\\mathbf{\\hat{k}_{N}}'
    assert latex(s) == '3 \\mathbf{{y}_{C}} \\mathbf{{x}_{N}}^{2}'
    assert latex(d[0]) == '(\\mathbf{\\hat{0}}|\\mathbf{\\hat{0}})'
    assert latex(d[4]) == '\\left(a\\right)\\left(\\mathbf{\\hat{i}_{N}}{\\middle|}' + '\\mathbf{\\hat{k}_{N}}\\right)'
    assert latex(d[9]) == '\\left(\\mathbf{\\hat{k}_{C}}{\\middle|}' + '\\mathbf{\\hat{k}_{N}}\\right) + \\left(' + '\\mathbf{\\hat{i}_{N}}{\\middle|}\\mathbf{' + '\\hat{k}_{N}}\\right)'
    assert latex(d[11]) == '\\left(a^{2} + b\\right)\\left(\\mathbf{\\hat{i}_{N}}' + '{\\middle|}\\mathbf{\\hat{k}_{N}}\\right) + ' + '\\left(\\int f{\\left(b \\right)}\\, db\\right)\\left(' + '\\mathbf{\\hat{k}_{N}}{\\middle|}\\mathbf{' + '\\hat{k}_{N}}\\right)'