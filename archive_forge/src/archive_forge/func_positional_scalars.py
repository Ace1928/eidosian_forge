from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
@property
def positional_scalars(self) -> List[LazyArgument]:
    return self.filtered_args(positional=True, keyword=False, values=False, scalars=True)