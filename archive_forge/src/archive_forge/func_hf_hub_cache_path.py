from __future__ import absolute_import
import pathlib
import contextlib
from typing import Optional, Union, Dict, TYPE_CHECKING
from lazyops.types.models import BaseSettings
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
from lazyops.imports._aiokeydb import (
@lazyproperty
def hf_hub_cache_path(self) -> FileLike:
    if self.huggingface_hub_cache:
        return File(self.huggingface_hub_cache) if _fileio_available else pathlib.Path(self.huggingface_hub_cache)
    return self.hf_home_path