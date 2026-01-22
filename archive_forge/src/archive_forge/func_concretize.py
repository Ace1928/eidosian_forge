import uuid
from typing import Generic, TypeVar, Optional
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from .measurements import MeasurementProcess, MidMeasure
def concretize(self, measurements: dict):
    """Returns a concrete value from a dictionary of hashes with concrete values."""
    values = tuple((measurements[meas] for meas in self.measurements))
    return self.processing_fn(*values)