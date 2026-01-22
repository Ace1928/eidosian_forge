from typing import List, Optional, Sequence, Set, Union
from torchgen import local
from torchgen.api.types import (
from torchgen.model import (
from torchgen.utils import assert_never
from .types import (
def default_expr(d: str, t: Type) -> str:
    if d == 'None' and str(t) == 'Tensor?':
        return '{}'
    if isinstance(t, BaseType) and t.name is BaseTy.str:
        if len(d) >= 2 and d[0] == "'" and (d[-1] == "'"):
            s = ''
            i = 1
            while i + 1 < len(d):
                if d[i] != '\\':
                    if d[i] == '"':
                        s += '\\"'
                    else:
                        s += d[i]
                    i += 1
                else:
                    if d[i + 1] == "'":
                        s += "'"
                    else:
                        s += d[i:i + 2]
                    i += 2
            return f'"{s}"'
    if isinstance(t, OptionalType):
        if d == 'None':
            return 'torch::executor::nullopt'
        return default_expr(d, t.elem)
    if isinstance(t, ListType):
        if d.startswith('[') and d.endswith(']'):
            return '{' + d[1:-1] + '}'
        elif t.size is None:
            raise ValueError(f"Expected a list default '[...]' but found: '{d}'")
    return JIT_TO_CPP_DEFAULT.get(d, d)