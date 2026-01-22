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
def upload_from(self, source: Union[str, os.PathLike], force_overwrite_to_cloud: bool=False) -> Self:
    """Upload a file or directory to the cloud path."""
    source = Path(source)
    if source.is_dir():
        for p in source.iterdir():
            (self / p.name).upload_from(p, force_overwrite_to_cloud=force_overwrite_to_cloud)
        return self
    else:
        if self.exists() and self.is_dir():
            dst = self / source.name
        else:
            dst = self
        dst._upload_file_to_cloud(source, force_overwrite_to_cloud=force_overwrite_to_cloud)
        return dst