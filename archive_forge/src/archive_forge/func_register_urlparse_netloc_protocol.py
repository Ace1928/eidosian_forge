import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def register_urlparse_netloc_protocol(protocol):
    """Ensure that protocol is setup to be used with urlparse netloc parsing."""
    if protocol not in urlutils.urlparse.uses_netloc:
        urlutils.urlparse.uses_netloc.append(protocol)