from typing import List
import datetime
import pytest
import numpy as np
import sympy
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor, VALID_LANGUAGES
def test_additional_validation():
    proc = SimulatedLocalProcessor(processor_id='test_proc', device=cirq_google.Sycamore23, validator=_no_y_gates)
    q = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.Y(q), cirq.measure(q, key='m'))
    sweep = cirq.Points(key='t', points=[1, 0])
    job = proc.run_sweep(circuit, params=sweep, repetitions=100)
    with pytest.raises(ValueError, match='No Y gates allowed!'):
        job.results()
    with pytest.raises(ValueError, match='No Y gates allowed!'):
        _ = proc.get_sampler().run_sweep(circuit, params=sweep, repetitions=100)