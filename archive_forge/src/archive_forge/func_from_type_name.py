from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
from monty.functools import lazy_property
from monty.json import MSONable
from pymatgen.core.libxcfunc import LibxcFunc
@classmethod
def from_type_name(cls, typ: str | None, name: str) -> Self:
    """Build the object from (type, name)."""
    for k, nt in cls.defined_aliases.items():
        if typ is not None and typ != nt.type:
            continue
        if name == nt.name:
            if len(k) == 1:
                return cls(xc=k)
            if len(k) == 2:
                return cls(x=k[0], c=k[1])
            raise ValueError(f'Wrong key: {k}')
    if '+' in name:
        x, c = (s.strip() for s in name.split('+'))
        return cls(x=LibxcFunc[x], c=LibxcFunc[c])
    return cls(xc=LibxcFunc[name])