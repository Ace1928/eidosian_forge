from typing import Any, FrozenSet, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import measurement_key_protocol
from cirq.type_workarounds import NotImplementedType
def measurement_keys_touched(val: Any) -> FrozenSet['cirq.MeasurementKey']:
    """Returns all the measurement keys used by the value.

    This would be the case if the value is or contains a measurement gate, or
    if the value is or contains a conditional operation.

    Args:
        val: The object that may interact with measurements.

    Returns:
        The measurement keys used by the value..
    """
    return measurement_key_protocol.measurement_key_objs(val) | control_keys(val)