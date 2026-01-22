from typing import Any, Dict, Optional, Tuple
from ..types import FloatsXd
from ..util import get_array_module
def inc_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
    key = (model_id, name)
    if self.proxy is not None:
        self.proxy.inc_grad(model_id, name, value)
    elif not self.has_grad(model_id, name):
        if hasattr(value, 'copy'):
            self._grads[key] = value.copy()
        elif not value.flags['C_CONTIGUOUS']:
            xp = get_array_module(value)
            self._grads[model_id, name] = xp.ascontiguousarray(value)
        else:
            self._grads[model_id, name] = value
    else:
        self._grads[model_id, name] += value