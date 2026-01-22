import io
import struct
from qiskit.qpy import formats
def write_mapping(file_obj, mapping, serializer, **kwargs):
    """Write a mapping in the file like object.

    .. note::

        This function must be used to make a binary data of mapping
        which include QPY serialized values.
        It's easier to use JSON serializer followed by encoding for standard data formats.
        This only supports flat dictionary and key must be string.

    Args:
        file_obj (File): A file like object to write data.
        mapping (Mapping): Object to serialize.
        serializer (Callable): Serializer callback that can handle mapping item.
            This must return type key and binary data of the mapping value.
        kwargs: Options set to the serializer.
    """
    num_elements = len(mapping)
    file_obj.write(struct.pack(formats.SEQUENCE_PACK, num_elements))
    for key, datum in mapping.items():
        key_bytes = key.encode(ENCODE)
        type_key, datum_bytes = serializer(datum, **kwargs)
        item_header = struct.pack(formats.MAP_ITEM_PACK, len(key_bytes), type_key, len(datum_bytes))
        file_obj.write(item_header)
        file_obj.write(key_bytes)
        file_obj.write(datum_bytes)