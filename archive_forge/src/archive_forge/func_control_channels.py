import re
import copy
import numbers
from typing import Dict, List, Any, Iterable, Tuple, Union
from collections import defaultdict
from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse.channels import (
@property
def control_channels(self) -> Dict[Tuple[int, ...], List]:
    """Return the control channels"""
    return self._control_channels