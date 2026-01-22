from __future__ import annotations
from typing import Any, Dict, Optional
def set_map(self, amap: Dict[Any, Any]):
    self._dispatch_map = amap
    return self