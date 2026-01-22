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
def create_noiseless_virtual_engine_from_proto(processor_ids: Union[str, List[str]], device_specifications: Union[v2.device_pb2.DeviceSpecification, List[v2.device_pb2.DeviceSpecification]]) -> SimulatedLocalEngine:
    """Creates a noiseless virtual engine object from a device specification proto.

    The device specification protocol buffer specifies qubits and gates on the device
    and can be retrieved from a stored "proto.txt" file or from the QCS API.

    Args:
        processor_ids: names of the processors to simulate.  These are arbitrary
            string identifiers and do not have to match the processors' names
            in QCS.  This can be a single string or list of strings.
        device_specifications:  `v2.device_pb2.DeviceSpecification` proto to create
            validating devices from.  This can be a single DeviceSpecification
            or a list of them.  There should be one DeviceSpecification for each
            processor_id.
        gate_sets: Iterable of serializers to use in the processor.

    Raises:
        ValueError: if processor_ids and device_specifications are not the same length.
    """
    if isinstance(processor_ids, str):
        processor_ids = [processor_ids]
    if isinstance(device_specifications, v2.device_pb2.DeviceSpecification):
        device_specifications = [device_specifications]
    if len(processor_ids) != len(device_specifications):
        raise ValueError('Must provide equal numbers of processor ids and device specifications.')
    return SimulatedLocalEngine(processors=[create_noiseless_virtual_processor_from_proto(processor_id, device_spec) for device_spec, processor_id in zip(device_specifications, processor_ids)])