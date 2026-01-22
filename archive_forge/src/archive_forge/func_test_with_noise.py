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
def test_with_noise():

    class Noise(cirq.NoiseModel):

        def noisy_operation(self, operation):
            yield operation
            if cirq.LineQubit(0) in operation.qubits:
                yield cirq.H(cirq.LineQubit(0))
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(q0), cirq.Y(q1), cirq.Z(q1), cirq.Moment([cirq.X(q0)]))
    c_expected = cirq.Circuit([cirq.Moment([cirq.X(q0), cirq.Y(q1)]), cirq.Moment([cirq.H(q0)]), cirq.Moment([cirq.Z(q1)]), cirq.Moment([cirq.X(q0)]), cirq.Moment([cirq.H(q0)])])
    c_noisy = c.with_noise(Noise())
    assert c_noisy == c_expected
    assert c.with_noise(None) == c
    assert c.with_noise(cirq.depolarize(0.1)) == cirq.Circuit(cirq.X(q0), cirq.Y(q1), cirq.Moment([d.with_tags(ops.VirtualTag()) for d in cirq.depolarize(0.1).on_each(q0, q1)]), cirq.Z(q1), cirq.Moment([d.with_tags(ops.VirtualTag()) for d in cirq.depolarize(0.1).on_each(q0, q1)]), cirq.Moment([cirq.X(q0)]), cirq.Moment([d.with_tags(ops.VirtualTag()) for d in cirq.depolarize(0.1).on_each(q0, q1)]))