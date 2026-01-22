import logging
import os.path
def get_supported_compression_types():
    """Return the list of supported compression types available to open.

    See compression paratemeter to smart_open.open().
    """
    return [NO_COMPRESSION, INFER_FROM_EXTENSION] + get_supported_extensions()