from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
@property
def target_mime_types(self) -> List[str]:
    """
        Return the target mime types
        """
    return [f'application/{t}' for t in self.targets] if self.targets else None