from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
def return_from_mutable_noop_redispatch(f: NativeFunction, inner_return_var: str) -> str:
    aliased, non_aliased = get_mutable_redispatch_return_names(f, inner_return_var)
    return return_str(f.func.returns, aliased + non_aliased)