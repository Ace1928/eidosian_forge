import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
@property
def params_dict(self) -> Dict[str, TuningParameterExpression]:
    """Get all tuning parameter expressions in depth-first order,
        with correspondent made-up new keys p0, p1, p2, ...
        """
    return {f'p{i}': x for i, x in enumerate(self.params)}