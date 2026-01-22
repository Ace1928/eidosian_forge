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
def download_to(self, destination: Union[str, os.PathLike]) -> Path:
    destination = Path(destination)
    if self.is_file():
        if destination.is_dir():
            destination = destination / self.name
        return self.client._download_file(self, destination)
    else:
        destination.mkdir(exist_ok=True)
        for f in self.iterdir():
            rel = str(self)
            if not rel.endswith('/'):
                rel = rel + '/'
            rel_dest = str(f)[len(rel):]
            f.download_to(destination / rel_dest)
        return destination