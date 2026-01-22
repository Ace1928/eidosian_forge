from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
def filtered_args(self, positional: bool=True, keyword: bool=True, values: bool=True, scalars: bool=True, generator: bool=True) -> List[LazyArgument]:
    args: List[LazyArgument] = []
    if positional:
        args.extend(self.positional_args)
    if keyword:
        args.extend(self.keyword_args)
    if values and scalars and generator:
        return args
    elif values and scalars:
        return [a for a in args if not a.is_generator]
    elif values:
        return [a for a in args if a.is_lazy_value]
    elif scalars:
        return [a for a in args if not a.is_lazy_value and (generator or not a.is_generator)]
    return []