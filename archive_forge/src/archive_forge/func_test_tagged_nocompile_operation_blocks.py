import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
def test_tagged_nocompile_operation_blocks():
    q = cirq.NamedQubit('q')
    u = cirq.Z(q).with_tags('nocompile')
    assert_optimizes(before=cirq.Circuit([cirq.Moment([cirq.Z(q)]), cirq.Moment([u])]), expected=cirq.Circuit([cirq.Moment([cirq.Z(q)]), cirq.Moment([u])]), with_context=True)