import itertools
import pytest
import cirq
import cirq_google as cg
import numpy as np
def test_hqp_missing_placement():
    hqp = cg.HardcodedQubitPlacer({cirq.LineTopology(5): dict(enumerate(cirq.LineQubit.range(5)))})
    circuit = cirq.testing.random_circuit(cirq.LineQubit.range(5), n_moments=2, op_density=1)
    shared_rt_info = cg.SharedRuntimeInfo(run_id='example')
    rs = np.random.RandomState(10)
    placed_c, _ = hqp.place_circuit(circuit, problem_topology=cirq.LineTopology(5), shared_rt_info=shared_rt_info, rs=rs)
    assert isinstance(placed_c, cirq.AbstractCircuit)
    circuit = cirq.testing.random_circuit(cirq.LineQubit.range(6), n_moments=2, op_density=1)
    with pytest.raises(cg.CouldNotPlaceError):
        hqp.place_circuit(circuit, problem_topology=cirq.LineTopology(6), shared_rt_info=shared_rt_info, rs=rs)