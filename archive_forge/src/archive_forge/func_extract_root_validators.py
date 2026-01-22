import warnings
from collections import ChainMap
from functools import partial, partialmethod, wraps
from itertools import chain
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, overload
from .errors import ConfigError
from .typing import AnyCallable
from .utils import ROOT_KEY, in_ipython
def extract_root_validators(namespace: Dict[str, Any]) -> Tuple[List[AnyCallable], List[Tuple[bool, AnyCallable]]]:
    from inspect import signature
    pre_validators: List[AnyCallable] = []
    post_validators: List[Tuple[bool, AnyCallable]] = []
    for name, value in namespace.items():
        validator_config: Optional[Validator] = getattr(value, ROOT_VALIDATOR_CONFIG_KEY, None)
        if validator_config:
            sig = signature(validator_config.func)
            args = list(sig.parameters.keys())
            if args[0] == 'self':
                raise ConfigError(f'Invalid signature for root validator {name}: {sig}, "self" not permitted as first argument, should be: (cls, values).')
            if len(args) != 2:
                raise ConfigError(f'Invalid signature for root validator {name}: {sig}, should be: (cls, values).')
            if validator_config.pre:
                pre_validators.append(validator_config.func)
            else:
                post_validators.append((validator_config.skip_on_failure, validator_config.func))
    return (pre_validators, post_validators)