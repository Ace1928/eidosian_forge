from __future__ import annotations
from typing import Any, Optional
@property
def ok_for_legacy(self) -> bool:
    """Return ``True`` if this read concern is compatible with
        old wire protocol versions.
        """
    return self.level is None or self.level == 'local'