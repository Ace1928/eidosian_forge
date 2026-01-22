import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def follow_redirection(transport, e, redirection_notice):
    mutter(redirection_notice)
    base, filename = urlutils.split(e.target)
    redirected_transport = get_transport(base)
    return redirected_transport