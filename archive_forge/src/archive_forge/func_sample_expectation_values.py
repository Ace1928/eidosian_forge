import collections
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import duet
import pandas as pd
from cirq import ops, protocols, study, value
from cirq.work.observable_measurement import (
from cirq.work.observable_settings import _hashable_param
def sample_expectation_values(self, program: 'cirq.AbstractCircuit', observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']], *, num_samples: int, params: 'cirq.Sweepable'=None, permit_terminal_measurements: bool=False) -> Sequence[Sequence[float]]:
    """Calculates estimated expectation values from samples of a circuit.

        Please see also `cirq.work.observable_measurement.measure_observables`
        for more control over how to measure a suite of observables.

        This method can be run on any device or simulator that supports circuit sampling. Compare
        with `simulate_expectation_values` in simulator.py, which is limited to simulators
        but provides exact results.

        Args:
            program: The circuit which prepares a state from which we sample expectation values.
            observables: A list of observables for which to calculate expectation values.
            num_samples: The number of samples to take. Increasing this value increases the
                statistical accuracy of the estimate.
            params: Parameters to run with the program.
            permit_terminal_measurements: If the provided circuit ends in a measurement, this
                method will generate an error unless this is set to True. This is meant to
                prevent measurements from ruining expectation value calculations.

        Returns:
            A list of expectation-value lists. The outer index determines the sweep, and the inner
            index determines the observable. For instance, results[1][3] would select the fourth
            observable measured in the second sweep.

        Raises:
            ValueError: If the number of samples was not positive, if empty observables were
                supplied, or if the provided circuit has terminal measurements and
                `permit_terminal_measurements` is true.
        """
    if num_samples <= 0:
        raise ValueError(f'Expectation values require at least one sample. Received: {num_samples}.')
    if not observables:
        raise ValueError('At least one observable must be provided.')
    if not permit_terminal_measurements and program.are_any_measurements_terminal():
        raise ValueError('Provided circuit has terminal measurements, which may skew expectation values. If this is intentional, set permit_terminal_measurements=True.')
    pauli_sums: List['cirq.PauliSum'] = [ops.PauliSum.wrap(o) for o in observables] if isinstance(observables, List) else [ops.PauliSum.wrap(observables)]
    del observables
    flat_pstrings: List['cirq.PauliString'] = []
    pstring_to_psum_i: Dict['cirq.PauliString', int] = {}
    for psum_i, pauli_sum in enumerate(pauli_sums):
        for pstring in pauli_sum:
            flat_pstrings.append(pstring)
            pstring_to_psum_i[pstring] = psum_i
    flat_params: List['cirq.ParamMappingType'] = [pr.param_dict for pr in study.to_resolvers(params)]
    circuit_param_to_sweep_i: Dict[FrozenSet[Tuple[str, Union[int, Tuple[int, int]]]], int] = {_hashable_param(param.items()): i for i, param in enumerate(flat_params)}
    obs_meas_results = measure_observables(circuit=program, observables=flat_pstrings, sampler=self, stopping_criteria=RepetitionsStoppingCriteria(total_repetitions=num_samples), readout_symmetrization=False, circuit_sweep=params, checkpoint=CheckpointFileOptions(checkpoint=False))
    nested_results: List[List[float]] = [[0] * len(pauli_sums) for _ in range(len(flat_params))]
    for res in obs_meas_results:
        param_i = circuit_param_to_sweep_i[_hashable_param(res.circuit_params.items())]
        psum_i = pstring_to_psum_i[res.setting.observable]
        nested_results[param_i][psum_i] += res.mean
    return nested_results