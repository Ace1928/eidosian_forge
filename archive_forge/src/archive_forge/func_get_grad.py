from typing import Any, Dict, Optional, Tuple
from ..types import FloatsXd
from ..util import get_array_module
def get_grad(self, model_id: int, name: str) -> FloatsXd:
    key = (model_id, name)
    return self._grads[key]