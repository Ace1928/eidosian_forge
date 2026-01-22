import io
import struct
from qiskit.qpy import formats
def write_type_key(file_obj, type_key):
    """Write a type key in the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        type_key (bytes): Type key to write.
    """
    file_obj.write(struct.pack('!1c', type_key))