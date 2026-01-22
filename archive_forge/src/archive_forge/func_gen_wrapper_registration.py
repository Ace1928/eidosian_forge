import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import (
from torchgen.model import (
from torchgen.utils import FileManager, mapMaybe
from .context import with_native_function_with_differentiability_info_and_key
from .gen_inplace_or_view_type import (
from .gen_trace_type import (
@with_native_function_and
def gen_wrapper_registration(f: NativeFunction, key: str='Default') -> str:
    return WRAPPER_REGISTRATION.substitute(unqual_operator_name_with_overload=f.func.name, type_wrapper_name=type_wrapper_name(f, key), class_type='VariableType')