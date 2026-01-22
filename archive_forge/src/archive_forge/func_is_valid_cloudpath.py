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
@classmethod
def is_valid_cloudpath(cls, path: Union[str, 'CloudPath'], raise_on_error: bool=False) -> Union[bool, TypeGuard[Self]]:
    valid = str(path).lower().startswith(cls.cloud_prefix.lower())
    if raise_on_error and (not valid):
        raise InvalidPrefixError(f"'{path}' is not a valid path since it does not start with '{cls.cloud_prefix}'")
    return valid