import itertools
import warnings
from math import log, pi
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._compiler import AbstractCompiler, QVMCompiler
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import QuantumComputer as QuantumComputerV3
from pyquil.api._quantum_computer import get_qc as get_qc_v3, QuantumExecutable
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.gates import MEASURE, RX
from pyquil.noise import NoiseModel, decoherence_noise_with_asymmetric_ro
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.quil import Program, validate_supported_quil
from pyquil.quilatom import qubit_index
from ._qam import StatefulQAM
def run_symmetrized_readout(self, program: Program, trials: int, symm_type: int=3, meas_qubits: Optional[List[int]]=None) -> np.ndarray:
    """
        Run a quil program in such a way that the readout error is made symmetric. Enforcing
        symmetric readout error is useful in simplifying the assumptions in some near
        term error mitigation strategies, see ``measure_observables`` for more information.

        The simplest example is for one qubit. In a noisy device, the probability of accurately
        reading the 0 state might be higher than that of the 1 state; due to e.g. amplitude
        damping. This makes correcting for readout more difficult. In the simplest case, this
        function runs the program normally ``(trials//2)`` times. The other half of the time,
        it will insert an ``X`` gate prior to any ``MEASURE`` instruction and then flip the
        measured classical bit back. Overall this has the effect of symmetrizing the readout error.

        The details. Consider preparing the input bitstring ``|i>`` (in the computational basis) and
        measuring in the Z basis. Then the Confusion matrix for the readout error is specified by
        the probabilities

             p(j|i) := Pr(measured = j | prepared = i ).

        In the case of a single qubit i,j \\in [0,1] then:
        there is no readout error if p(0|0) = p(1|1) = 1.
        the readout error is symmetric if p(0|0) = p(1|1) = 1 - epsilon.
        the readout error is asymmetric if p(0|0) != p(1|1).

        If your quantum computer has this kind of asymmetric readout error then
        ``qc.run_symmetrized_readout`` will symmetrize the readout error.

        The readout error above is only asymmetric on a single bit. In practice the confusion
        matrix on n bits need not be symmetric, e.g. for two qubits p(ij|ij) != 1 - epsilon for
        all i,j. In these situations a more sophisticated means of symmetrization is needed; and
        we use orthogonal arrays (OA) built from Hadamard matrices.

        The symmetrization types are specified by an int; the types available are:
        -1 -- exhaustive symmetrization uses every possible combination of flips
        0 -- trivial that is no symmetrization
        1 -- symmetrization using an OA with strength 1
        2 -- symmetrization using an OA with strength 2
        3 -- symmetrization using an OA with strength 3
        In the context of readout symmetrization the strength of the orthogonal array enforces
        the symmetry of the marginal confusion matrices.

        By default a strength 3 OA is used; this ensures expectations of the form
        ``<b_k . b_j . b_i>`` for bits any bits i,j,k will have symmetric readout errors. Here
        expectation of a random variable x as is denote ``<x> = sum_i Pr(i) x_i``. It turns out that
        a strength 3 OA is also a strength 2 and strength 1 OA it also ensures ``<b_j . b_i>`` and
        ``<b_i>`` have symmetric readout errors for any bits b_j and b_i.

        :param program: The program to run symmetrized readout on.
        :param trials: The minimum number of times to run the program; it is recommend that this
            number should be in the hundreds or thousands. This parameter will be mutated if
            necessary.
        :param symm_type: the type of symmetrization
        :param meas_qubits: An advanced feature. The groups of measurement qubits. Only these
            qubits will be symmetrized over, even if the program acts on other qubits.
        :return: A numpy array of shape (trials, len(ro-register)) that contains 0s and 1s.
        """
    if not isinstance(symm_type, int):
        raise ValueError('Symmetrization options are indicated by an int. See the docstrings for more information.')
    if meas_qubits is None:
        meas_qubits = list(cast(Set[int], program.get_qubits()))
    trials = _check_min_num_trials_for_symmetrized_readout(len(meas_qubits), trials, symm_type)
    sym_programs, flip_arrays = _symmetrization(program, meas_qubits, symm_type)
    num_shots_per_prog = trials // len(sym_programs)
    if num_shots_per_prog * len(sym_programs) < trials:
        warnings.warn(f'The number of trials was modified from {trials} to {num_shots_per_prog * len(sym_programs)}. To be consistent with the number of trials required by the type of readout symmetrization chosen.')
    results = _measure_bitstrings(self, sym_programs, meas_qubits, num_shots_per_prog)
    return _consolidate_symmetrization_outputs(results, flip_arrays)