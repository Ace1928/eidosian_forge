from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
@property
def lazy_type(self) -> CType:
    assert self.lazy_type_ is not None, f'Attempted to access lazy_type for invalid argument {self.name}'
    return self.lazy_type_