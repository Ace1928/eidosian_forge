import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def run_zeta_chi_gamma_compensation_for_moments(circuit: cirq.Circuit, sampler: Union[AbstractEngine, cirq.Sampler], processor_id: Optional[str]=None, options: FloquetPhasedFSimCalibrationOptions=THETA_ZETA_GAMMA_FLOQUET_PHASED_FSIM_CHARACTERIZATION, gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]=try_convert_syc_or_sqrt_iswap_to_fsim, merge_subsets: bool=True, max_layers_per_request: int=1, progress_func: Optional[Callable[[int, int], None]]=None, permit_mixed_moments: bool=False) -> Tuple[CircuitWithCalibration, List[PhasedFSimCalibrationResult]]:
    """Compensates circuit for errors in zeta, chi and gamma angles by running on the engine.

    The method calls prepare_floquet_characterization_for_moments to extract moments to
    characterize, run_calibrations to characterize them and
    make_zeta_chi_gamma_compensation_for_moments to compensate the circuit with characterization
    data.

    Args:
        circuit: Circuit to characterize and calibrate.
        sampler: cirq_google.Engine or cirq.Sampler object used for running the calibrations. When
            sampler is cirq_google.Engine or cirq_google.ProcessorSampler object then the
            calibrations are issued against a Google's quantum device. The only other sampler
            supported for simulation purposes is cirq_google.PhasedFSimEngineSimulator.
        processor_id: Used when sampler is cirq_google.Engine object and passed to
            cirq_google.Engine.run_calibrations method.
        options: Options that are applied to each characterized gate within a moment. Defaults
            to all_except_for_chi_options which is the broadest currently supported choice.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.
        merge_subsets: Whether to merge moments that can be characterized at the same time
            together.
        max_layers_per_request: Maximum number of calibration requests issued to cirq.Engine at a
            single time. Defaults to 1.
        progress_func: Optional callback function that might be used to report the calibration
            progress. The callback is called with two integers, the first one being a number of
            layers already calibrated and the second one the total number of layers to calibrate.
        permit_mixed_moments: Whether to allow mixing single-qubit and two-qubit gates in a single
            moment.

    Returns:
        Tuple of:
          - Calibrated circuit together with its calibration metadata in CircuitWithCalibration
            object. The calibrated circuit has single-qubit Z gates added which compensates for the
            true gates imperfections.
            The moment to calibration mapping is updated for the new circuit so that successive
            calibrations could be applied.
          - List of characterizations results that were obtained in order to calibrate the circuit.
    """
    circuit_with_calibration, requests = prepare_floquet_characterization_for_moments(circuit, options, gates_translator, merge_subsets=merge_subsets, permit_mixed_moments=permit_mixed_moments)
    characterizations = run_calibrations(calibrations=requests, sampler=sampler, processor_id=processor_id, max_layers_per_request=max_layers_per_request, progress_func=progress_func)
    calibrated_circuit = make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, characterizations, gates_translator=gates_translator, permit_mixed_moments=permit_mixed_moments)
    return (calibrated_circuit, characterizations)