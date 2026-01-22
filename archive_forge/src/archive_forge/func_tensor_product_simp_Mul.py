from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sympify import sympify
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.matrixutils import (
from sympy.physics.quantum.trace import Tr
def tensor_product_simp_Mul(e):
    """Simplify a Mul with TensorProducts.

    Current the main use of this is to simplify a ``Mul`` of ``TensorProduct``s
    to a ``TensorProduct`` of ``Muls``. It currently only works for relatively
    simple cases where the initial ``Mul`` only has scalars and raw
    ``TensorProduct``s, not ``Add``, ``Pow``, ``Commutator``s of
    ``TensorProduct``s.

    Parameters
    ==========

    e : Expr
        A ``Mul`` of ``TensorProduct``s to be simplified.

    Returns
    =======

    e : Expr
        A ``TensorProduct`` of ``Mul``s.

    Examples
    ========

    This is an example of the type of simplification that this function
    performs::

        >>> from sympy.physics.quantum.tensorproduct import                     tensor_product_simp_Mul, TensorProduct
        >>> from sympy import Symbol
        >>> A = Symbol('A',commutative=False)
        >>> B = Symbol('B',commutative=False)
        >>> C = Symbol('C',commutative=False)
        >>> D = Symbol('D',commutative=False)
        >>> e = TensorProduct(A,B)*TensorProduct(C,D)
        >>> e
        AxB*CxD
        >>> tensor_product_simp_Mul(e)
        (A*C)x(B*D)

    """
    if not isinstance(e, Mul):
        return e
    c_part, nc_part = e.args_cnc()
    n_nc = len(nc_part)
    if n_nc == 0:
        return e
    elif n_nc == 1:
        if isinstance(nc_part[0], Pow):
            return Mul(*c_part) * tensor_product_simp_Pow(nc_part[0])
        return e
    elif e.has(TensorProduct):
        current = nc_part[0]
        if not isinstance(current, TensorProduct):
            if isinstance(current, Pow):
                if isinstance(current.base, TensorProduct):
                    current = tensor_product_simp_Pow(current)
            else:
                raise TypeError('TensorProduct expected, got: %r' % current)
        n_terms = len(current.args)
        new_args = list(current.args)
        for next in nc_part[1:]:
            if isinstance(next, TensorProduct):
                if n_terms != len(next.args):
                    raise QuantumError('TensorProducts of different lengths: %r and %r' % (current, next))
                for i in range(len(new_args)):
                    new_args[i] = new_args[i] * next.args[i]
            elif isinstance(next, Pow):
                if isinstance(next.base, TensorProduct):
                    new_tp = tensor_product_simp_Pow(next)
                    for i in range(len(new_args)):
                        new_args[i] = new_args[i] * new_tp.args[i]
                else:
                    raise TypeError('TensorProduct expected, got: %r' % next)
            else:
                raise TypeError('TensorProduct expected, got: %r' % next)
            current = next
        return Mul(*c_part) * TensorProduct(*new_args)
    elif e.has(Pow):
        new_args = [tensor_product_simp_Pow(nc) for nc in nc_part]
        return tensor_product_simp_Mul(Mul(*c_part) * TensorProduct(*new_args))
    else:
        return e