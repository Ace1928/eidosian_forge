from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
def is_valid_source_path(self, path: Optional[InputPathType]=None, **kwargs) -> bool:
    """
        Validate whether the path is valid

        path: /path/to/file.pdf
        """
    if path is None:
        return False
    if not isinstance(path, str) and hasattr(path, 'suffix'):
        return path.suffix == self.source_ext
    return str(path).endswith(self.source_ext)