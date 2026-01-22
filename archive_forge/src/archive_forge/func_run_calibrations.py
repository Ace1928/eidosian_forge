import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def run_calibrations(calibrations: Sequence[PhasedFSimCalibrationRequest], sampler: Union[AbstractEngine, cirq.Sampler], processor_id: Optional[str]=None, max_layers_per_request: int=1, progress_func: Optional[Callable[[int, int], None]]=None) -> List[PhasedFSimCalibrationResult]:
    """Runs calibration requests on the Engine.

    Args:
        calibrations: List of calibrations to perform described in a request object.
        sampler: cirq_google.Engine or cirq.Sampler object used for running the calibrations. When
            sampler is cirq_google.Engine or cirq_google.ProcessorSampler object then the
            calibrations are issued against a Google's quantum device. The only other sampler
            supported for simulation purposes is cirq_google.PhasedFSimEngineSimulator.
        processor_id: Used when sampler is cirq_google.Engine object and passed to
            cirq_google.Engine.run_calibrations method.
        max_layers_per_request: Maximum number of calibration requests issued to cirq.Engine at a
            single time. Defaults to 1.
        progress_func: Optional callback function that might be used to report the calibration
            progress. The callback is called with two integers, the first one being a number of
            layers already calibrated and the second one the total number of layers to calibrate.

    Returns:
        List of PhasedFSimCalibrationResult for each requested calibration.

    Raises:
        ValueError: If less than one layers was requested to be calibrated, if calibrations of
            different types was supplied, if no `processor_id` or `gate_set` is provided, or
            if the calibration / sampler combo is not supported.
    """
    if max_layers_per_request < 1:
        raise ValueError(f'Maximum number of layers per request must be at least 1, {max_layers_per_request} given')
    if not calibrations:
        return []
    calibration_request_types = set((type(cr) for cr in calibrations))
    if len(calibration_request_types) > 1:
        raise ValueError(f'All calibrations must be of the same type. You gave: {calibration_request_types}')
    calibration_request_type, = calibration_request_types
    if isinstance(sampler, AbstractEngine):
        if processor_id is None:
            raise ValueError('processor_id must be provided.')
        processor: Optional[AbstractProcessor] = sampler.get_processor(processor_id=processor_id)
    elif isinstance(sampler, ProcessorSampler):
        processor = sampler.processor
    else:
        processor = None
    if processor is not None:
        if calibration_request_type == LocalXEBPhasedFSimCalibrationRequest:
            engine_sampler = processor.get_sampler()
            return _run_local_calibrations_via_sampler(calibrations, engine_sampler)
        return _run_calibrations_via_engine(calibrations, processor, max_layers_per_request, progress_func)
    if calibration_request_type == LocalXEBPhasedFSimCalibrationRequest:
        return _run_local_calibrations_via_sampler(calibrations, sampler=cast(cirq.Sampler, sampler))
    if isinstance(sampler, PhasedFSimEngineSimulator):
        return sampler.get_calibrations(calibrations)
    raise ValueError(f'Unsupported sampler/request combination: Sampler {sampler} cannot run calibration request of type {calibration_request_type}')