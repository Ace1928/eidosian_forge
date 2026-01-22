from typing import cast, Dict, Hashable, Iterable, List, Optional, Sequence
from collections import OrderedDict
import dataclasses
import numpy as np
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import result_pb2
def results_to_proto(trial_sweeps: Iterable[Iterable[cirq.Result]], measurements: List[MeasureInfo], *, out: Optional[result_pb2.Result]=None) -> result_pb2.Result:
    """Converts trial results from multiple sweeps to v2 protobuf message.

    Args:
        trial_sweeps: Iterable over sweeps and then over trial results within
            each sweep.
        measurements: List of info about measurements in the program.
        out: Optional message to populate. If not given, create a new message.

    Raises:
        ValueError: If the number of repetitions in trial results were not all the same.
    """
    if out is None:
        out = result_pb2.Result()
    for trial_sweep in trial_sweeps:
        sweep_result = out.sweep_results.add()
        for i, trial_result in enumerate(trial_sweep):
            if i == 0:
                sweep_result.repetitions = trial_result.repetitions
            elif trial_result.repetitions != sweep_result.repetitions:
                raise ValueError('Different numbers of repetitions in one sweep.')
            reps = sweep_result.repetitions
            pr = sweep_result.parameterized_results.add()
            pr.params.assignments.update(trial_result.params.param_dict)
            for m in measurements:
                mr = pr.measurement_results.add()
                mr.key = m.key
                mr.instances = m.instances
                m_data = trial_result.records[m.key]
                for i, qubit in enumerate(m.qubits):
                    qmr = mr.qubit_measurement_results.add()
                    qmr.qubit.id = v2.qubit_to_proto_id(qubit)
                    qmr.results = pack_bits(m_data[:, :, i].reshape(reps * m.instances))
    return out