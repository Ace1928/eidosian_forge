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
def get_meta_data(self, index=None):
    """get_meta_data(index=None)

            Read meta data from the file. using the image index. If the
            index is omitted or None, return the file's (global) meta data.

            Note that ``get_data`` also provides the meta data for the returned
            image as an attribute of that image.

            The meta data is a dict, which shape depends on the format.
            E.g. for JPEG, the dict maps group names to subdicts and each
            group is a dict with name-value pairs. The groups represent
            the different metadata formats (EXIF, XMP, etc.).
            """
    self._checkClosed()
    meta = self._get_meta_data(index)
    if not isinstance(meta, dict):
        raise ValueError('Meta data must be a dict, not %r' % meta.__class__.__name__)
    return meta