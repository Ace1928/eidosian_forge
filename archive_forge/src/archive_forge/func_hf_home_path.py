from __future__ import absolute_import
import pathlib
import contextlib
from typing import Optional, Union, Dict, TYPE_CHECKING
from lazyops.types.models import BaseSettings
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
from lazyops.imports._aiokeydb import (
@lazyproperty
def hf_home_path(self) -> FileLike:
    hf_home = self.hf_home or self.model_path.joinpath('.hf')
    return File(hf_home) if _fileio_available else pathlib.Path(hf_home)