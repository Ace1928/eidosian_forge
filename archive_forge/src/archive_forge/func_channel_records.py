import abc
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq.value import digits, value_equality_attr
@property
def channel_records(self) -> Mapping['cirq.MeasurementKey', List[int]]:
    """Gets the a mapping from measurement key to channel measurement records."""
    return self._channel_records