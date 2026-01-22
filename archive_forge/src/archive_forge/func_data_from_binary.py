import io
import struct
from qiskit.qpy import formats
def data_from_binary(binary_data, deserializer, **kwargs):
    """Load object from binary data with specified deserializer.

    Args:
        binary_data (bytes): Binary data to deserialize.
        deserializer (Callable): Deserializer callback that can handle input object type.
        kwargs: Options set to the deserializer.

    Returns:
        any: Deserialized object.
    """
    with io.BytesIO(binary_data) as container:
        container.seek(0)
        obj = deserializer(container, **kwargs)
    return obj