import gc
from . import msgpack
from .msgpack import msgpack_encoders, msgpack_decoders  # noqa: F401
from .util import force_path, FilePath, JSONInputBin, JSONOutputBin
def msgpack_dumps(data: JSONInputBin) -> bytes:
    """Serialize an object to a msgpack byte string.

    data: The data to serialize.
    RETURNS (bytes): The serialized bytes.
    """
    return msgpack.dumps(data, use_bin_type=True)