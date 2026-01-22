import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def prepare_characterization_for_operations(circuit: Union[cirq.Circuit, Iterable[cirq.Circuit]], options: PhasedFSimCalibrationOptions[RequestT], *, gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]=try_convert_syc_or_sqrt_iswap_to_fsim, permit_mixed_moments: bool=False) -> List[RequestT]:
    """Extracts a minimal set of characterization requests necessary to characterize all the
    operations within a circuit(s).

    This prepare method works on two-qubit operations of the circuit. The method extracts
    all the operations and groups them in a way to minimize the number of characterizations
    requested, depending on the connectivity.

    Contrary to prepare_characterization_for_moments, this method ignores moments structure
    and is less accurate because certain errors caused by cross-talk are ignored.

    The major advantage of this method is that the number of generated characterization requests is
    bounded by four for grid-like devices, where for
    prepare_characterization_for_moments the number of characterizations is bounded by
    number of moments in a circuit.

    The circuit can only be composed of single qubit operations, wait operations, measurement
    operations and operations supported by gates_translator.

    Args:
        circuit: Circuit or circuits to characterize. Only circuits with qubits of type GridQubit
            that can be covered by HALF_GRID_STAGGERED_PATTERN are supported
        options: Options that are applied to each characterized gate within a moment.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        List of PhasedFSimCalibrationRequest for each group of operations to characterize.

    Raises:
        IncompatibleMomentError: When circuit contains a moment with operations other than the
            operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
        ValueError: If unable to cover all interactions with a half grid staggered pattern.
    """
    circuits = [circuit] if isinstance(circuit, cirq.Circuit) else circuit
    pairs, gate = _extract_all_pairs_to_characterize(circuits, gates_translator, permit_mixed_moments)
    if gate is None:
        return []
    characterizations = []
    for pattern in HALF_GRID_STAGGERED_PATTERN:
        pattern_pairs = [pair for pair in pairs if pair in pattern]
        if pattern_pairs:
            characterizations.append(options.create_phased_fsim_request(pairs=tuple(sorted(pattern_pairs)), gate=gate))
    if sum((len(characterization.pairs) for characterization in characterizations)) != len(pairs):
        raise ValueError('Unable to cover all interactions with HALF_GRID_STAGGERED_PATTERN')
    return characterizations