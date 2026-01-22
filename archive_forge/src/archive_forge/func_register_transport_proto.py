import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def register_transport_proto(prefix, help=None, info=None, register_netloc=False):
    transport_list_registry.register_transport(prefix, help)
    if register_netloc:
        if not prefix.endswith('://'):
            raise ValueError(prefix)
        register_urlparse_netloc_protocol(prefix[:-3])