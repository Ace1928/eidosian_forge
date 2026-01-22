from __future__ import annotations
import os
from typing import Any, Collection, cast
def running_in_sis() -> bool:
    """Return whether this app is running in SiS."""
    try:
        from snowflake.snowpark._internal.utils import is_in_stored_procedure
        return cast(bool, is_in_stored_procedure())
    except ModuleNotFoundError:
        return False