import warnings
from collections import ChainMap
from functools import partial, partialmethod, wraps
from itertools import chain
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, overload
from .errors import ConfigError
from .typing import AnyCallable
from .utils import ROOT_KEY, in_ipython
def check_for_unused(self) -> None:
    unused_validators = set(chain.from_iterable(((getattr(v.func, '__name__', f'<No __name__: id:{id(v.func)}>') for v in self.validators[f] if v.check_fields) for f in self.validators.keys() - self.used_validators)))
    if unused_validators:
        fn = ', '.join(unused_validators)
        raise ConfigError(f"Validators defined with incorrect fields: {fn} (use check_fields=False if you're inheriting from the model and intended this)")