from typing import FrozenSet, Mapping, Optional, Tuple
import dataclasses
@classmethod
def parse_serialized(cls, key_str: str) -> 'MeasurementKey':
    """Parses the serialized string representation of `Measurementkey` into a `MeasurementKey`.

        This is the only way to construct a `MeasurementKey` from a nested string representation
        (where the path is joined to the key name by the `MEASUREMENT_KEY_SEPARATOR`)"""
    components = key_str.split(MEASUREMENT_KEY_SEPARATOR)
    return MeasurementKey(name=components[-1], path=tuple(components[:-1]))