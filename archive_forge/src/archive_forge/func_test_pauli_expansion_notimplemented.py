import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_pauli_expansion_notimplemented():
    assert cirq.IdentityGate(1, (3,))._pauli_expansion_() == NotImplemented