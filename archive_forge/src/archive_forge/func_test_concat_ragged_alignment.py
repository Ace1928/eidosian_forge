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
def test_concat_ragged_alignment():
    a, b = cirq.LineQubit.range(2)
    assert cirq.Circuit.concat_ragged(cirq.Circuit(cirq.X(a)), cirq.Circuit(cirq.Y(b)) * 4, cirq.Circuit(cirq.Z(a)), align='first') == cirq.Circuit(cirq.Moment(cirq.X(a), cirq.Y(b)), cirq.Moment(cirq.Y(b)), cirq.Moment(cirq.Y(b)), cirq.Moment(cirq.Z(a), cirq.Y(b)))
    assert cirq.Circuit.concat_ragged(cirq.Circuit(cirq.X(a)), cirq.Circuit(cirq.Y(b)) * 4, cirq.Circuit(cirq.Z(a)), align='left') == cirq.Circuit(cirq.Moment(cirq.X(a), cirq.Y(b)), cirq.Moment(cirq.Z(a), cirq.Y(b)), cirq.Moment(cirq.Y(b)), cirq.Moment(cirq.Y(b)))
    assert cirq.Circuit.concat_ragged(cirq.Circuit(cirq.X(a)), cirq.Circuit(cirq.Y(b)) * 4, cirq.Circuit(cirq.Z(a)), align='right') == cirq.Circuit(cirq.Moment(cirq.Y(b)), cirq.Moment(cirq.Y(b)), cirq.Moment(cirq.Y(b)), cirq.Moment(cirq.X(a), cirq.Y(b)), cirq.Moment(cirq.Z(a)))