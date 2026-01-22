import io
import struct
from qiskit.qpy import formats
def mapping_to_binary(mapping, serializer, **kwargs):
    """Convert mapping into binary data with specified serializer.

    .. note::

        This function must be used to make a binary data of mapping
        which include QPY serialized values.
        It's easier to use JSON serializer followed by encoding for standard data formats.
        This only supports flat dictionary and key must be string.

    Args:
        mapping (Mapping): Object to serialize.
        serializer (Callable): Serializer callback that can handle mapping item.
            This must return type key and binary data of the mapping value.
        kwargs: Options set to the serializer.

    Returns:
        bytes: Binary data.
    """
    with io.BytesIO() as container:
        write_mapping(container, mapping, serializer, **kwargs)
        binary_data = container.getvalue()
    return binary_data