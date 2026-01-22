from pathlib import Path
import numpy as np
from ..config import known_extensions
from .request import InitializationError, IOMode
from .v3_plugin_api import ImageProperties, PluginV3
def legacy_get_writer(self, **kwargs):
    """legacy_get_writer(**kwargs)

        Returns a :class:`.Writer` object which can be used to write data
        and meta data to the specified file.

        Parameters
        ----------
        kwargs : ...
            Further keyword arguments are passed to the writer. See :func:`.help`
            to see what arguments are available for a particular format.
        """
    self._request._kwargs = kwargs
    return self._format.get_writer(self._request)