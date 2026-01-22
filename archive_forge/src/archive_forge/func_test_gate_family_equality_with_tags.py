from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
@pytest.mark.parametrize('tags_to_accept_fam1, tags_to_ignore_fam1, tags_to_accept_fam2, tags_to_ignore_fam2', [(tuple('ab'), tuple('cd'), tuple('ba'), tuple('dc')), (tuple('ab'), [], tuple('ba'), []), ([], tuple('ab'), [], tuple('ba'))])
def test_gate_family_equality_with_tags(tags_to_accept_fam1, tags_to_ignore_fam1, tags_to_accept_fam2, tags_to_ignore_fam2):
    gate_fam1 = cirq.GateFamily(cirq.X, tags_to_accept=tags_to_accept_fam1, tags_to_ignore=tags_to_ignore_fam1)
    gate_fam2 = cirq.GateFamily(cirq.X, tags_to_accept=tags_to_accept_fam2, tags_to_ignore=tags_to_ignore_fam2)
    assert gate_fam1 == gate_fam2