from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
def test_gateset_contains_with_tags():
    tag = 'PhysicalZTag'
    gf_accept = cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[tag])
    gf_ignore = cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[tag])
    op = cirq.Z(q)
    op_with_tag = cirq.Z(q).with_tags(tag)
    assert op in cirq.Gateset(gf_ignore)
    assert op_with_tag not in cirq.Gateset(gf_ignore)
    assert op not in cirq.Gateset(gf_accept)
    assert op_with_tag in cirq.Gateset(gf_accept)
    assert op in cirq.Gateset(gf_accept, gf_ignore)
    assert op_with_tag in cirq.Gateset(gf_accept, gf_ignore)