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
def create_noiseless_virtual_engine_from_templates(processor_ids: Union[str, List[str]], template_names: Union[str, List[str]]) -> SimulatedLocalEngine:
    """Creates a noiseless virtual engine object from a device specification template.

    Args:
        processor_ids: names of the processors to simulate.  These are arbitrary
            string identifiers and do not have to match the processors' names
            in QCS.  There can be a single string or a list of strings for multiple
            processors.
        template_names: File names of the device specification templates, see
            cirq_google/devices/specifications for valid templates.  There can
            be a single str for a template name or a list of strings.  Each
            template name should be matched to a single processor id.
        gate_sets: Iterable of serializers to use in the processor.

    Raises:
        ValueError: if processor_ids and template_names are not the same length.
    """
    if isinstance(processor_ids, str):
        processor_ids = [processor_ids]
    if isinstance(template_names, str):
        template_names = [template_names]
    if len(processor_ids) != len(template_names):
        raise ValueError('Must provide equal numbers of processor ids and template names.')
    specifications = [_create_device_spec_from_template(template_name) for template_name in template_names]
    return create_noiseless_virtual_engine_from_proto(processor_ids, specifications)