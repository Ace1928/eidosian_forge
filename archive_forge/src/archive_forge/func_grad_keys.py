from typing import Any, Dict, Optional, Tuple
from ..types import FloatsXd
from ..util import get_array_module
@property
def grad_keys(self) -> Tuple[KeyT, ...]:
    return tuple([key for key in self.param_keys if self.has_grad(*key)])