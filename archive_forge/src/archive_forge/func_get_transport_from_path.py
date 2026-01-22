import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def get_transport_from_path(path, possible_transports=None):
    """Open a transport for a local path.

    :param path: Local path as byte or unicode string
    :return: Transport object for path
    """
    return get_transport_from_url(urlutils.local_path_to_url(path), possible_transports)