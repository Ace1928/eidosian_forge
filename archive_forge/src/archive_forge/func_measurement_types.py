import abc
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq.value import digits, value_equality_attr
@property
def measurement_types(self) -> Mapping['cirq.MeasurementKey', 'cirq.MeasurementType']:
    """Gets the a mapping from measurement key to the measurement type."""
    return self._measurement_types