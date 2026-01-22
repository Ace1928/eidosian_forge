import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
def is_reference_for_foreach(f: NativeFunction, function_schema: FunctionSchema) -> bool:
    return f.func.name.name.base.split('_foreach_')[-1] == function_schema.name.name.base and (not function_schema.name.name.inplace or str(f.func.name) in _foreach_with_inplace_ref) and all((ref_arg.type in (arg.type, getattr(arg.type, 'elem', None)) for arg, ref_arg in zip(f.func.arguments.flat_non_out, function_schema.arguments.flat_non_out)))