import inspect
from functools import partial
from typing import (
def auto_rich_repr(self: Type[T]) -> Result:
    """Auto generate __rich_rep__ from signature of __init__"""
    try:
        signature = inspect.signature(self.__init__)
        for name, param in signature.parameters.items():
            if param.kind == param.POSITIONAL_ONLY:
                yield getattr(self, name)
            elif param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                if param.default == param.empty:
                    yield getattr(self, param.name)
                else:
                    yield (param.name, getattr(self, param.name), param.default)
    except Exception as error:
        raise ReprError(f'Failed to auto generate __rich_repr__; {error}') from None