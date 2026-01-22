import sys
import warnings
import contextlib
import numpy as np
from pathlib import Path
from . import Array, asarray
from .request import ImageMode
from ..config import known_plugins, known_extensions, PluginConfig, FileExtension
from ..config.plugins import _original_order
from .imopen import imopen
def set_meta_data(self, meta):
    """set_meta_data(meta)

            Sets the file's (global) meta data. The meta data is a dict which
            shape depends on the format. E.g. for JPEG the dict maps
            group names to subdicts, and each group is a dict with
            name-value pairs. The groups represents the different
            metadata formats (EXIF, XMP, etc.).

            Note that some meta formats may not be supported for
            writing, and individual fields may be ignored without
            warning if they are invalid.
            """
    self._checkClosed()
    if not isinstance(meta, dict):
        raise ValueError('Meta must be a dict.')
    else:
        return self._set_meta_data(meta)