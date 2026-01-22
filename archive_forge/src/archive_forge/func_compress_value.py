from __future__ import annotations
import abc
import zlib
import hashlib
from lazyops.types import BaseModel
from lazyops.utils.logs import logger
from lazyops.utils.pooler import ThreadPooler
from typing import Any, Optional, Union, Dict, TypeVar, TYPE_CHECKING
def compress_value(self, value: Union[str, bytes], **kwargs) -> Union[str, bytes]:
    """
        Compresses the value
        """
    if self.compression_enabled:
        if isinstance(value, str):
            value = value.encode(self.encoding)
        return self.compressor.compress(value)
    return value