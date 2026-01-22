from typing import Collection, Dict, Optional, List, Set, Tuple, cast
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import device_pb2
from cirq_google.devices import grid_device
from cirq_google.experimental.ops import coupler_pulse
from cirq_google.ops import physical_z_tag, sycamore_gate
Populates `DeviceSpecification.valid_targets` with the device's qubit pairs.

    Args:
        pairs: The collection of the device's bi-directional qubit pairs.
        out: The `DeviceSpecification` to be populated.
    