import re
import warnings
from numbers import Number
from pathlib import Path
from typing import Dict
import numpy as np
from imageio.core.legacy_plugin_wrapper import LegacyPlugin
from imageio.core.util import Array
from imageio.core.v3_plugin_api import PluginV3
from . import formats
from .config import known_extensions, known_plugins
from .core import RETURN_BYTES
from .core.imopen import imopen
def volread(uri, format=None, **kwargs):
    """volread(uri, format=None, **kwargs)

    Reads a volume from the specified file. Returns a numpy array, which
    comes with a dict of meta data at its 'meta' attribute.

    Parameters
    ----------
    uri : {str, pathlib.Path, bytes, file}
        The resource to load the volume from, e.g. a filename, pathlib.Path,
        http address or file object, see the docs for more info.
    format : str
        The format to use to read the file. By default imageio selects
        the appropriate for you based on the filename and its contents.
    kwargs : ...
        Further keyword arguments are passed to the reader. See :func:`.help`
        to see what arguments are available for a particular format.
    """
    imopen_args = decypher_format_arg(format)
    imopen_args['legacy_mode'] = True
    with imopen(uri, 'rv', **imopen_args) as file:
        return file.read(index=0, **kwargs)