import itertools
import pytest
import cirq
import cirq_google as cg
import numpy as np
def test_random_device_placer_bad_device():
    topo = cirq.LineTopology(8)
    qubits = cirq.LineQubit.range(8)
    circuit = cirq.testing.random_circuit(qubits, n_moments=8, op_density=1.0, random_state=52)
    qp = cg.RandomDevicePlacer()
    with pytest.raises(ValueError, match='.*shared_rt_info\\.device.*'):
        qp.place_circuit(circuit, problem_topology=topo, shared_rt_info=cg.SharedRuntimeInfo(run_id='1'), rs=np.random.RandomState(1))