import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def make_zeta_chi_gamma_compensation_for_operations(circuit: cirq.Circuit, characterizations: List[PhasedFSimCalibrationResult], gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]=try_convert_gate_to_fsim, permit_mixed_moments: bool=False) -> cirq.Circuit:
    """Compensates circuit operations against errors in zeta, chi and gamma angles.

    This method creates a new circuit with a single-qubit Z gates added in a such way so that
    zeta, chi and gamma angles discovered by characterizations are cancelled-out and set to 0.

    Contrary to make_zeta_chi_gamma_compensation_for_moments this method does not match
    characterizations to the moment structure of the circuits and thus is less accurate because
    some errors caused by cross-talks are not mitigated.

    The major advantage of this method over make_zeta_chi_gamma_compensation_for_moments is that it
    can work with arbitrary set of characterizations that cover all the interactions of the circuit
    (up to assumptions of merge_matching_results method). In particular, for grid-like devices the
    number of characterizations is bounded by four, where in the case of
    make_zeta_chi_gamma_compensation_for_moments the number of characterizations is bounded by
    number of moments in a circuit.

    This function preserves a moment structure of the circuit. All single qubit gates appear on new
    moments in the final circuit.

    Args:
        circuit: Circuit to calibrate.
        characterizations: List of characterization results (likely returned from run_calibrations).
            All the characterizations must be compatible in sense of merge_matching_results, they
            will be merged together.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        permit_mixed_moments: Whether to allow mixing single-qubit and two-qubit gates in a single
            moment.

    Returns:
        Calibrated circuit with a single-qubit Z gates added which compensates for the true gates
        imperfections.
    """
    characterization = merge_matching_results(characterizations)
    moment_to_calibration = [0] * len(circuit)
    calibrated = _make_zeta_chi_gamma_compensation(CircuitWithCalibration(circuit, moment_to_calibration), [characterization] if characterization is not None else [], gates_translator, permit_mixed_moments=permit_mixed_moments)
    return calibrated.circuit