from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.symbol import Symbol
from sympy.physics.quantum import TensorProduct
def evaluate_pauli_product(arg):
    """Help function to evaluate Pauli matrices product
    with symbolic objects.

    Parameters
    ==========

    arg: symbolic expression that contains Paulimatrices

    Examples
    ========

    >>> from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
    >>> from sympy import I
    >>> evaluate_pauli_product(I*Pauli(1)*Pauli(2))
    -sigma3

    >>> from sympy.abc import x
    >>> evaluate_pauli_product(x**2*Pauli(2)*Pauli(1))
    -I*x**2*sigma3
    """
    start = arg
    end = arg
    if isinstance(arg, Pow) and isinstance(arg.args[0], Pauli):
        if arg.args[1].is_odd:
            return arg.args[0]
        else:
            return 1
    if isinstance(arg, Add):
        return Add(*[evaluate_pauli_product(part) for part in arg.args])
    if isinstance(arg, TensorProduct):
        return TensorProduct(*[evaluate_pauli_product(part) for part in arg.args])
    elif not isinstance(arg, Mul):
        return arg
    while not start == end or (start == arg and end == arg):
        start = end
        tmp = start.as_coeff_mul()
        sigma_product = 1
        com_product = 1
        keeper = 1
        for el in tmp[1]:
            if isinstance(el, Pauli):
                sigma_product *= el
            elif not el.is_commutative:
                if isinstance(el, Pow) and isinstance(el.args[0], Pauli):
                    if el.args[1].is_odd:
                        sigma_product *= el.args[0]
                elif isinstance(el, TensorProduct):
                    keeper = keeper * sigma_product * TensorProduct(*[evaluate_pauli_product(part) for part in el.args])
                    sigma_product = 1
                else:
                    keeper = keeper * sigma_product * el
                    sigma_product = 1
            else:
                com_product *= el
        end = tmp[0] * keeper * sigma_product * com_product
        if end == arg:
            break
    return end