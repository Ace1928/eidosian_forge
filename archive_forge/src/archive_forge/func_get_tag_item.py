from contextlib import ExitStack
import logging
import os
import warnings
from collections import OrderedDict
from fiona import compat, vfs
from fiona.ogrext import Iterator, ItemsIterator, KeysIterator
from fiona.ogrext import Session, WritingSession
from fiona.ogrext import buffer_to_virtual_file, remove_virtual_file, GEOMETRY_TYPES
from fiona.errors import (
from fiona.logutils import FieldSkipLogFilter
from fiona.crs import CRS
from fiona._env import get_gdal_release_name, get_gdal_version_tuple
from fiona.env import env_ctx_if_needed
from fiona.errors import FionaDeprecationWarning
from fiona.drvsupport import (
from fiona.path import Path, vsi_path, parse_path
def get_tag_item(self, key, ns=None):
    """Returns tag item value

        Parameters
        ----------
        key: str
            The key for the metadata item to fetch.
        ns: str, optional
            Used to select a namespace other than the default.

        Returns
        -------
        str
        """
    if _GDAL_VERSION_TUPLE.major < 2:
        raise GDALVersionError('get_tag_item requires GDAL 2+, fiona was compiled against: {}'.format(_GDAL_RELEASE_NAME))
    if self.session:
        return self.session.get_tag_item(key=key, ns=ns)
    return None