from typing import Any, overload, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def definitely_commutes(v1: Any, v2: Any, *, atol: Union[int, float]=1e-08) -> bool:
    """Determines whether two values definitely commute.

    Returns:
        True: The two values definitely commute.
        False: The two values may or may not commute.
    """
    return commutes(v1, v2, atol=atol, default=False)