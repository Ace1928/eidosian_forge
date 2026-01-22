from __future__ import annotations
import types
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import NumbaUtilError
def maybe_use_numba(engine: str | None) -> bool:
    """Signal whether to use numba routines."""
    return engine == 'numba' or (engine is None and GLOBAL_USE_NUMBA)