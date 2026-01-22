import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_to_text_diagram_teleportation_to_diagram(circuit_cls):
    ali = cirq.NamedQubit('(0, 0)')
    bob = cirq.NamedQubit('(0, 1)')
    msg = cirq.NamedQubit('(1, 0)')
    tmp = cirq.NamedQubit('(1, 1)')
    c = circuit_cls([cirq.Moment([cirq.H(ali)]), cirq.Moment([cirq.CNOT(ali, bob)]), cirq.Moment([cirq.X(msg) ** 0.5]), cirq.Moment([cirq.CNOT(msg, ali)]), cirq.Moment([cirq.H(msg)]), cirq.Moment([cirq.measure(msg), cirq.measure(ali)]), cirq.Moment([cirq.CNOT(ali, bob)]), cirq.Moment([cirq.CNOT(msg, tmp)]), cirq.Moment([cirq.CZ(bob, tmp)])])
    cirq.testing.assert_has_diagram(c, '\n(0, 0): ───H───@───────────X───────M───@───────────\n               │           │           │\n(0, 1): ───────X───────────┼───────────X───────@───\n                           │                   │\n(1, 0): ───────────X^0.5───@───H───M───────@───┼───\n                                           │   │\n(1, 1): ───────────────────────────────────X───@───\n')
    cirq.testing.assert_has_diagram(c, '\n(0, 0): ---H---@-----------X-------M---@-----------\n               |           |           |\n(0, 1): -------X-----------|-----------X-------@---\n                           |                   |\n(1, 0): -----------X^0.5---@---H---M-------@---|---\n                                           |   |\n(1, 1): -----------------------------------X---@---\n', use_unicode_characters=False)
    cirq.testing.assert_has_diagram(c, '\n(0, 0) (0, 1) (1, 0) (1, 1)\n|      |      |      |\nH      |      |      |\n|      |      |      |\n@------X      |      |\n|      |      |      |\n|      |      X^0.5  |\n|      |      |      |\nX-------------@      |\n|      |      |      |\n|      |      H      |\n|      |      |      |\nM      |      M      |\n|      |      |      |\n@------X      |      |\n|      |      |      |\n|      |      @------X\n|      |      |      |\n|      @-------------@\n|      |      |      |\n', use_unicode_characters=False, transpose=True)