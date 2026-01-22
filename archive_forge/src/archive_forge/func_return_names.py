from typing import List, Optional, Sequence, Set, Union
from torchgen import local
from torchgen.api.types import (
from torchgen.model import (
from torchgen.utils import assert_never
from .types import (
def return_names(f: NativeFunction, *, fallback_name: str='result') -> Sequence[str]:
    returns: List[str] = []
    for i, r in enumerate(f.func.returns):
        if f.func.name.name.inplace:
            assert i == 0, 'illegal inplace function with multiple returns'
            name = 'self'
        elif f.func.is_out_fn():
            name = f.func.arguments.out[i].name
        elif r.name:
            name_conflict = any((r.name == a.name for a in f.func.schema_order_arguments()))
            if name_conflict and (not f.func.is_out_fn()):
                name = f'{r.name}_return'
            else:
                name = r.name
        else:
            name = fallback_name if len(f.func.returns) == 1 else f'{fallback_name}{i}'
        returns.append(name)
    return returns