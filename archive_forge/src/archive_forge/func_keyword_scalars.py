from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
@property
def keyword_scalars(self) -> List[LazyArgument]:
    return self.filtered_args(positional=False, keyword=True, values=False, scalars=True)