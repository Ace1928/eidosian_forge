from functools import wraps
from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript
@wraps(batch_derivatives)
def compute_derivatives(self, circuits, execution_config=DefaultExecutionConfig):
    is_single_circuit = False
    if isinstance(circuits, QuantumScript):
        is_single_circuit = True
        circuits = (circuits,)
    jacs = batch_derivatives(self, circuits, execution_config)
    return jacs[0] if is_single_circuit else jacs