import concurrent
import os
import time
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
import numpy as np
import pandas as pd
import tqdm
from cirq import ops, devices, value, protocols
from cirq.circuits import Circuit, Moment
from cirq.experiments.random_quantum_circuit_generation import CircuitLibraryCombination
def sample_2q_xeb_circuits(sampler: 'cirq.Sampler', circuits: Sequence['cirq.Circuit'], cycle_depths: Sequence[int], *, repetitions: int=10000, batch_size: int=9, progress_bar: Optional[Callable[..., ContextManager]]=tqdm.tqdm, combinations_by_layer: Optional[List[CircuitLibraryCombination]]=None, shuffle: Optional['cirq.RANDOM_STATE_OR_SEED_LIKE']=None, dataset_directory: Optional[str]=None):
    """Sample two-qubit XEB circuits given a sampler.

    Args:
        sampler: A Cirq sampler for executing circuits.
        circuits: A library of two-qubit circuits generated from
            `random_rotations_between_two_qubit_circuit` of sufficient length for `cycle_depths`.
        cycle_depths: A sequence of cylce depths at which we will truncate each of the `circuits`
            to execute.
        repetitions: Each (circuit, cycle_depth) will be sampled for this many repetitions.
        batch_size: We call `run_batch` on the sampler, which can speed up execution in certain
            environments. The number of (circuit, cycle_depth) tasks to be run in each batch
            is given by this number.
        progress_bar: A progress context manager following the `tqdm` API or `None` to not report
            progress.
        combinations_by_layer: Either `None` or the result of
            `rqcg.get_random_combinations_for_device`. If this is `None`, the circuits specified
            by `circuits` will be sampled verbatim, resulting in isolated XEB characterization.
            Otherwise, this contains all the random combinations and metadata required to combine
            the circuits in `circuits` into wide, parallel-XEB-style circuits for execution.
        shuffle: If provided, use this random state or seed to shuffle the order in which tasks
            are executed.
        dataset_directory: If provided, save each batch of sampled results to a file
            `{dataset_directory}/xeb.{uuid4()}.json` where uuid4() is a random string. This can be
            used to incrementally save results to be analyzed later.

    Returns:
        A pandas dataframe with index given by ['circuit_i', 'cycle_depth'].
        Columns always include "sampled_probs". If `combinations_by_layer` is
        not `None` and you are doing parallel XEB, additional metadata columns
        will be attached to the returned DataFrame.
    """
    if progress_bar is None:
        progress_bar = _NoProgress
    if combinations_by_layer is None:
        combinations_by_layer, circuits = _get_combinations_by_layer_for_isolated_xeb(circuits)
        one_pair = True
    else:
        _verify_two_line_qubits_from_circuits(circuits)
        one_pair = False
    zipped_circuits = _zip_circuits(circuits, combinations_by_layer)
    tasks = _generate_sample_2q_xeb_tasks(zipped_circuits, cycle_depths)
    if shuffle is not None:
        shuffle = value.parse_random_state(shuffle)
        shuffle.shuffle(tasks)
    records = _execute_sample_2q_xeb_tasks_in_batches(tasks=tasks, sampler=sampler, combinations_by_layer=combinations_by_layer, repetitions=repetitions, batch_size=batch_size, progress_bar=progress_bar, dataset_directory=dataset_directory)
    df = pd.DataFrame(records).set_index(['circuit_i', 'cycle_depth'])
    if one_pair:
        df = df.drop(['layer_i', 'pair_i', 'combination_i'], axis=1)
    return df