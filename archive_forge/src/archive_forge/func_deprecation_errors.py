from __future__ import annotations
from typing import Any, MutableMapping, Optional
@property
def deprecation_errors(self) -> Optional[bool]:
    """The API deprecation errors setting.

        When set, this value is sent to the server in the
        "apiDeprecationErrors" field.
        """
    return self._deprecation_errors