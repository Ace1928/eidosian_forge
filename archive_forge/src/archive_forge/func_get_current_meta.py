import traceback
from contextlib import contextmanager
from typing import List, Any, Dict
from ._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def get_current_meta() -> Dict[str, Any]:
    return current_meta