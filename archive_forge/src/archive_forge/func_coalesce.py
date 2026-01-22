import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
def coalesce(*arg: Any) -> Any:
    """Return the first non-none value in the list of arguments.

    Similar to ?? in C#.
    """
    return next((a for a in arg if a is not None), None)