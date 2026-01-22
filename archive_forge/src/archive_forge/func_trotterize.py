import re
from itertools import product
import numpy as np
import copy
from typing import (
from pyquil.quilatom import (
from .quil import Program
from .gates import H, RZ, RX, CNOT, X, PHASE, QUANTUM_GATES
from numbers import Number, Complex
from collections import OrderedDict
import warnings
def trotterize(first_pauli_term: PauliTerm, second_pauli_term: PauliTerm, trotter_order: int=1, trotter_steps: int=1) -> Program:
    """
    Create a Quil program that approximates exp( (A + B)t) where A and B are
    PauliTerm operators.

    :param first_pauli_term: PauliTerm denoted `A`
    :param second_pauli_term: PauliTerm denoted `B`
    :param trotter_order: Optional argument indicating the Suzuki-Trotter
                          approximation order--only accepts orders 1, 2, 3, 4.
    :param trotter_steps: Optional argument indicating the number of products
                          to decompose the exponential into.

    :return: Quil program
    """
    if not 1 <= trotter_order < 5:
        raise ValueError('trotterize only accepts trotter_order in {1, 2, 3, 4}.')
    commutator = first_pauli_term * second_pauli_term + -1 * second_pauli_term * first_pauli_term
    prog = Program()
    if is_zero(commutator):
        param_exp_prog_one = exponential_map(first_pauli_term)
        exp_prog = param_exp_prog_one(1)
        prog += exp_prog
        param_exp_prog_two = exponential_map(second_pauli_term)
        exp_prog = param_exp_prog_two(1)
        prog += exp_prog
        return prog
    order_slices = suzuki_trotter(trotter_order, trotter_steps)
    for coeff, operator in order_slices:
        if operator == 0:
            param_prog = exponential_map(coeff * first_pauli_term)
            exp_prog = param_prog(1)
            prog += exp_prog
        else:
            param_prog = exponential_map(coeff * second_pauli_term)
            exp_prog = param_prog(1)
            prog += exp_prog
    return prog