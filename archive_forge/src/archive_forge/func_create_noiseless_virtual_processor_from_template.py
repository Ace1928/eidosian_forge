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
def create_noiseless_virtual_processor_from_template(processor_id: str, template_name: str) -> SimulatedLocalProcessor:
    """Creates a simulated local processor from a device specification template.

    Args:
        processor_id: name of the processor to simulate.  This is an arbitrary
            string identifier and does not have to match the processor's name
            in QCS.
        template_name: File name of the device specification template, see
            cirq_google/devices/specifications for valid templates.
        gate_sets: Iterable of serializers to use in the processor.
    """
    return create_noiseless_virtual_processor_from_proto(processor_id, device_specification=_create_device_spec_from_template(template_name))