from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
def get_mutable_redispatch_return_names(f: NativeFunction, inner_return_var: str) -> Tuple[List[str], List[str]]:
    aliased_returns = []
    non_aliased_returns = []
    for i, name in enumerate(f.func.aliased_return_names()):
        if name is not None:
            aliased_returns.append(name)
        else:
            non_aliased_returns.append(inner_return_var if len(f.func.returns) == 1 else f'std::get<{i}>({inner_return_var})')
    return (aliased_returns, non_aliased_returns)