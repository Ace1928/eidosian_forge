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
def create_noiseless_virtual_engine_from_device(processor_id: str, device: cirq.Device, device_specification: Optional[v2.device_pb2.DeviceSpecification]=None) -> SimulatedLocalEngine:
    """Creates an Engine object with a single processor backed by a noiseless simulator.

    Creates a noiseless engine object based on the cirq simulator,
    a default validator, and a provided device.

    Args:
        processor_id: name of the processor to simulate.  This is an arbitrary
            string identifier and does not have to match the processor's name
            in QCS.
        device: A `cirq.Device` to validate circuits against.
        device_specification: a` DeviceSpecification` proto that the processor
            should return if `get_device_specification()` is queried.
    """
    return SimulatedLocalEngine([_create_virtual_processor_from_device(processor_id, device, device_specification)])