import numpy as np
import pytest
import cirq
def test_projector_repr():
    q0 = cirq.NamedQubit('q0')
    assert repr(cirq.ProjectorString({q0: 0})) == "cirq.ProjectorString(projector_dict={cirq.NamedQubit('q0'): 0},coefficient=(1+0j))"