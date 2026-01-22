import abc
from collections import defaultdict
import collections.abc
from contextlib import contextmanager
import os
from pathlib import (  # type: ignore
import shutil
import sys
from typing import (
from urllib.parse import urlparse
from warnings import warn
from cloudpathlib.enums import FileCacheMode
from . import anypath
from .exceptions import (
def register_path_class(key: str) -> Callable[[Type[CloudPathT]], Type[CloudPathT]]:

    def decorator(cls: Type[CloudPathT]) -> Type[CloudPathT]:
        if not issubclass(cls, CloudPath):
            raise TypeError('Only subclasses of CloudPath can be registered.')
        implementation_registry[key]._path_class = cls
        cls._cloud_meta = implementation_registry[key]
        return cls
    return decorator