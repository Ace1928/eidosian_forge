import abc
import dataclasses
from functools import lru_cache
from typing import Dict, Any, Tuple, List, Callable, TYPE_CHECKING, Hashable
import numpy as np
import cirq
from cirq import _compat
from cirq.devices.named_topologies import get_placements, NamedTopology
from cirq.protocols import obj_to_dict_helper
from cirq_google.workflow._device_shim import _Device_dot_get_nx_graph
def place_circuit(self, circuit: 'cirq.AbstractCircuit', problem_topology: 'cirq.NamedTopology', shared_rt_info: 'cg.SharedRuntimeInfo', rs: np.random.RandomState) -> Tuple['cirq.FrozenCircuit', Dict[Any, 'cirq.Qid']]:
    """Place a circuit with a given topology onto a device via `cirq.get_placements` with
        randomized selection of the placement each time.

        This requires device information to be present in `shared_rt_info`.

        Args:
            circuit: The circuit.
            problem_topology: The topologies (i.e. connectivity) of the circuit.
            shared_rt_info: A `cg.SharedRuntimeInfo` object that contains a `device` attribute
                of type `cirq.Device` to enable placement.
            rs: A `RandomState` as a source of randomness for random placements.

        Returns:
            A tuple of a new frozen circuit with the qubits placed and a mapping from input
            qubits or nodes to output qubits.

        Raises:
            ValueError: If `shared_rt_info` does not have a device field.
        """
    device = shared_rt_info.device
    if device is None:
        raise ValueError('RandomDevicePlacer requires shared_rt_info.device to be a `cirq.Device`. This should have been set during the initialization phase of `cg.execute`.')
    placement = _get_random_placement(problem_topology, device, rs=rs, topo_node_to_qubit_func=self.topo_node_to_qubit_func)
    return (circuit.unfreeze().transform_qubits(placement).freeze(), placement)