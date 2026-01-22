import abc
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq.value import digits, value_equality_attr
def get_digits(self, key: 'cirq.MeasurementKey', index=-1) -> Tuple[int, ...]:
    return self._records[key][index] if self._measurement_types[key] == MeasurementType.MEASUREMENT else (self._channel_records[key][index],)