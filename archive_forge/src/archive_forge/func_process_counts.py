from typing import Sequence, Tuple
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from .measurements import Probability, SampleMeasurement, StateMeasurement
from .mid_measure import MeasurementValue
def process_counts(self, counts: dict, wire_order: Wires) -> np.ndarray:
    wire_map = dict(zip(wire_order, range(len(wire_order))))
    mapped_wires = [wire_map[w] for w in self.wires]
    if mapped_wires:
        mapped_counts = {}
        for outcome, occurrence in counts.items():
            mapped_outcome = ''.join((outcome[i] for i in mapped_wires))
            mapped_counts[mapped_outcome] = mapped_counts.get(mapped_outcome, 0) + occurrence
        counts = mapped_counts
    num_shots = sum(counts.values())
    num_wires = len(next(iter(counts)))
    dim = 2 ** num_wires
    prob_vector = qml.math.zeros(dim, dtype='float64')
    for outcome, occurrence in counts.items():
        prob_vector[int(outcome, base=2)] = occurrence / num_shots
    return prob_vector