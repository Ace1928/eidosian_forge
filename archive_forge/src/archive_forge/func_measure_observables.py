import abc
import dataclasses
import itertools
import os
import tempfile
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, study, ops, value, protocols
from cirq._doc import document
from cirq.work.observable_grouping import group_settings_greedy, GROUPER_T
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import InitObsSetting, observables_to_settings, _MeasurementSpec
def measure_observables(circuit: 'cirq.AbstractCircuit', observables: Iterable['cirq.PauliString'], sampler: Union['cirq.Simulator', 'cirq.Sampler'], stopping_criteria: StoppingCriteria, *, readout_symmetrization: bool=False, circuit_sweep: Optional['cirq.Sweepable']=None, grouper: Union[str, GROUPER_T]=group_settings_greedy, readout_calibrations: Optional[BitstringAccumulator]=None, checkpoint: CheckpointFileOptions=CheckpointFileOptions()) -> List[ObservableMeasuredResult]:
    """Measure a collection of PauliString observables for a state prepared by a Circuit.

    If you need more control over the process, please see `measure_grouped_settings` for a
    lower-level API. If you would like your results returned as a pandas DataFrame,
    please see `measure_observables_df`.

    Args:
        circuit: The circuit used to prepare the state to measure. This can contain parameters,
            in which case you should also specify `circuit_sweep`.
        observables: A collection of PauliString observables to measure. These will be grouped
            into simultaneously-measurable groups, see `grouper` argument.
        sampler: The sampler.
        stopping_criteria: A StoppingCriteria object to indicate how precisely to sample
            measurements for estimating observables.
        readout_symmetrization: If set to True, each run will be split into two: one normal and
            one where a bit flip is incorporated prior to measurement. In the latter case, the
            measured bit will be flipped back classically and accumulated together. This causes
            readout error to appear symmetric, p(0|0) = p(1|1).
        circuit_sweep: Additional parameter sweeps for parameters contained in `circuit`. The
            total sweep is the product of the circuit sweep with parameter settings for the
            single-qubit basis-change rotations.
        grouper: Either "greedy" or a function that groups lists of `InitObsSetting`. See the
            documentation for the `grouped_settings` argument of `measure_grouped_settings` for
            full details.
        readout_calibrations: The result of `calibrate_readout_error`.
        checkpoint: Options to set up optional checkpointing of intermediate data for each
            iteration of the sampling loop. See the documentation for `CheckpointFileOptions` for
            more. Load in these results with `cirq.read_json`.

    Returns:
        A list of ObservableMeasuredResult; one for each input PauliString.
    """
    qubits = _get_all_qubits(circuit, observables)
    settings = list(observables_to_settings(observables, qubits))
    actual_grouper = _parse_grouper(grouper)
    grouped_settings = actual_grouper(settings)
    accumulators = measure_grouped_settings(circuit=circuit, grouped_settings=grouped_settings, sampler=sampler, stopping_criteria=stopping_criteria, circuit_sweep=circuit_sweep, readout_symmetrization=readout_symmetrization, readout_calibrations=readout_calibrations, checkpoint=checkpoint)
    return flatten_grouped_results(accumulators)