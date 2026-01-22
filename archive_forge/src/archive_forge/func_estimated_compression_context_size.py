from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def estimated_compression_context_size(self):
    """Estimated size in bytes needed to compress with these parameters."""
    return lib.ZSTD_estimateCCtxSize_usingCCtxParams(self._params)