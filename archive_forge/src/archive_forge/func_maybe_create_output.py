from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
def maybe_create_output(f: NativeFunction, var_name: str) -> str:
    if len(f.func.returns) == 0:
        return ''
    return_type = dispatcher.returns_type(f.func.returns).remove_const_ref().cpp_type()
    return f'{return_type} {var_name} = '