from __future__ import annotations
from typing import Any
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.cg import CG, Wigner3j, Wigner6j, Wigner9j
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import CGate, CNotGate, IdentityGate, UGate, XGate
from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace, HilbertSpace, L2
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import Operator, OuterProduct, DifferentialOperator
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.qubit import Qubit, IntQubit
from sympy.physics.quantum.spin import Jz, J2, JzBra, JzBraCoupled, JzKet, JzKetCoupled, Rotation, WignerD
from sympy.physics.quantum.state import Bra, Ket, TimeDepBra, TimeDepKet
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.sho1d import RaisingOp
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices.dense import Matrix
from sympy.sets.sets import Interval
from sympy.testing.pytest import XFAIL
from sympy.physics.quantum.spin import JzOp
from sympy.printing import srepr
from sympy.printing.pretty import pretty as xpretty
from sympy.printing.latex import latex
def test_state():
    x = symbols('x')
    bra = Bra()
    ket = Ket()
    bra_tall = Bra(x / 2)
    ket_tall = Ket(x / 2)
    tbra = TimeDepBra()
    tket = TimeDepKet()
    assert str(bra) == '<psi|'
    assert pretty(bra) == '<psi|'
    assert upretty(bra) == '⟨ψ❘'
    assert latex(bra) == '{\\left\\langle \\psi\\right|}'
    sT(bra, "Bra(Symbol('psi'))")
    assert str(ket) == '|psi>'
    assert pretty(ket) == '|psi>'
    assert upretty(ket) == '❘ψ⟩'
    assert latex(ket) == '{\\left|\\psi\\right\\rangle }'
    sT(ket, "Ket(Symbol('psi'))")
    assert str(bra_tall) == '<x/2|'
    ascii_str = ' / |\n/ x|\n\\ -|\n \\2|'
    ucode_str = ' ╱ │\n╱ x│\n╲ ─│\n ╲2│'
    assert pretty(bra_tall) == ascii_str
    assert upretty(bra_tall) == ucode_str
    assert latex(bra_tall) == '{\\left\\langle \\frac{x}{2}\\right|}'
    sT(bra_tall, "Bra(Mul(Rational(1, 2), Symbol('x')))")
    assert str(ket_tall) == '|x/2>'
    ascii_str = '| \\ \n|x \\\n|- /\n|2/ '
    ucode_str = '│ ╲ \n│x ╲\n│─ ╱\n│2╱ '
    assert pretty(ket_tall) == ascii_str
    assert upretty(ket_tall) == ucode_str
    assert latex(ket_tall) == '{\\left|\\frac{x}{2}\\right\\rangle }'
    sT(ket_tall, "Ket(Mul(Rational(1, 2), Symbol('x')))")
    assert str(tbra) == '<psi;t|'
    assert pretty(tbra) == '<psi;t|'
    assert upretty(tbra) == '⟨ψ;t❘'
    assert latex(tbra) == '{\\left\\langle \\psi;t\\right|}'
    sT(tbra, "TimeDepBra(Symbol('psi'),Symbol('t'))")
    assert str(tket) == '|psi;t>'
    assert pretty(tket) == '|psi;t>'
    assert upretty(tket) == '❘ψ;t⟩'
    assert latex(tket) == '{\\left|\\psi;t\\right\\rangle }'
    sT(tket, "TimeDepKet(Symbol('psi'),Symbol('t'))")