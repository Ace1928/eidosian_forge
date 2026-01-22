import gc
from . import msgpack
from .msgpack import msgpack_encoders, msgpack_decoders  # noqa: F401
from .util import force_path, FilePath, JSONInputBin, JSONOutputBin
def write_msgpack(path: FilePath, data: JSONInputBin) -> None:
    """Create a msgpack file and dump contents.

    location (FilePath): The file path.
    data (JSONInputBin): The data to serialize.
    """
    file_path = force_path(path, require_exists=False)
    with file_path.open('wb') as f:
        msgpack.dump(data, f, use_bin_type=True)