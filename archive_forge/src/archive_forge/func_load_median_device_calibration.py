import json
from typing import cast, List, Optional, Union, Type
import pathlib
import time
import google.protobuf.text_format as text_format
import cirq
from cirq.sim.simulator import SimulatesSamples
from cirq_google.api import v2
from cirq_google.engine import calibration, engine_validator, simulated_local_processor, util
from cirq_google.devices import grid_device
from cirq_google.devices.google_noise_properties import NoiseModelFromGoogleNoiseProperties
from cirq_google.engine.calibration_to_noise_properties import noise_properties_from_calibration
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor
def load_median_device_calibration(processor_id: str) -> calibration.Calibration:
    """Loads a median `cirq_google.Calibration` for the given device.

    Real calibration data from Google's 'rainbow' and 'weber' devices has been
    saved in Cirq. The calibrations selected are roughly representative of the
    median performance for that chip.

    A description of the stored metrics can be found on the
    [calibration page](https://quantumai.google/cirq/google/calibration).

    Args:
        processor_id: name of the processor to simulate.

    Raises:
        ValueError: if processor_id is not a supported QCS processor.
    """
    cal_name = MEDIAN_CALIBRATIONS.get(processor_id, None)
    if cal_name is None:
        raise ValueError(f'Got processor_id={processor_id}, but no median calibration is defined for that processor.')
    path = pathlib.Path(__file__).parent.parent.resolve()
    with path.joinpath('devices', 'calibrations', cal_name).open() as f:
        cal = cast(calibration.Calibration, cirq.read_json(f))
    cal.timestamp = MEDIAN_CALIBRATION_TIMESTAMPS[processor_id]
    return cal