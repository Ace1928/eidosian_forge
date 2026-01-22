import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def register_transport_provider(self, key, obj):
    self.get(key).insert(0, registry._ObjectGetter(obj))