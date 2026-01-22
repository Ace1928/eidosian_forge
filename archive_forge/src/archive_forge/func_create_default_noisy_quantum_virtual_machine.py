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
def create_default_noisy_quantum_virtual_machine(processor_id: str, simulator_class: Optional[Type[SimulatesSamples]]=None, **kwargs) -> SimulatedLocalEngine:
    """Creates a virtual engine with a noisy simulator based on a processor id.

    Args:
        processor_id: The string name of a processor that has available noise data.
        simulator_class: The class of the type of simulator to be initialized. The
            simulator class initializer needs to support the `noise` parameter.
        **kwargs: Other arguments which are passed through to the simulator initializer.
            The 'noise' argument will be overwritten with a new noise model.

    Returns:
        A SimulatedLocalEngine that uses a simulator of type simulator_class with a
            noise model based on available noise data for the processor processor_id.
    """
    if simulator_class is None:
        try:
            import qsimcirq
            simulator_class = qsimcirq.QSimSimulator
        except ImportError:
            simulator_class = cirq.Simulator
    calibration = load_median_device_calibration(processor_id)
    noise_properties = noise_properties_from_calibration(calibration)
    noise_model = NoiseModelFromGoogleNoiseProperties(noise_properties)
    simulator = simulator_class(noise=noise_model, **kwargs)
    device_specification = create_device_spec_from_processor_id(processor_id)
    device = create_device_from_processor_id(processor_id)
    simulated_processor = SimulatedLocalProcessor(processor_id=processor_id, sampler=simulator, device=device, calibrations={calibration.timestamp // 1000: calibration}, device_specification=device_specification)
    return SimulatedLocalEngine([simulated_processor])