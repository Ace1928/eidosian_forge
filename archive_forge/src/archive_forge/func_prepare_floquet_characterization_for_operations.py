import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def prepare_floquet_characterization_for_operations(circuit: Union[cirq.Circuit, Iterable[cirq.Circuit]], options: FloquetPhasedFSimCalibrationOptions=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]=try_convert_syc_or_sqrt_iswap_to_fsim, permit_mixed_moments: bool=False) -> List[FloquetPhasedFSimCalibrationRequest]:
    """Extracts a minimal set of Floquet characterization requests necessary to characterize all the
    operations within a circuit(s).

    This variant of prepare method works on two-qubit operations of the circuit. The method extracts
    all the operations and groups them in a way to minimize the number of characterizations
    requested, depending on the connectivity.

    Contrary to prepare_floquet_characterization_for_moments, this method ignores moments structure
    and is less accurate because certain errors caused by cross-talks are ignored.

    The major advantage of this method is that the number of generated characterization requests is
    bounded by four for grid-like devices, where for the
    prepare_floquet_characterization_for_moments the number of characterizations is bounded by
    number of moments in a circuit.

    The circuit can only be composed of single qubit operations, wait operations, measurement
    operations and operations supported by gates_translator.

    Args:
        circuit: Circuit or circuits to characterize. Only circuits with qubits of type GridQubit
            that can be covered by HALF_GRID_STAGGERED_PATTERN are supported
        options: Options that are applied to each characterized gate within a moment. Defaults
            to all_except_for_chi_options which is the broadest currently supported choice.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        List of FloquetPhasedFSimCalibrationRequest for each group of operations to characterize.

    Raises:
        IncompatibleMomentError when circuit contains a moment with operations other than the
        operations matched by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    return prepare_characterization_for_operations(circuit=circuit, options=options, gates_translator=gates_translator, permit_mixed_moments=permit_mixed_moments)