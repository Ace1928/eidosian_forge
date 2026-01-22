import functools
import inspect
import itertools
import types
from contextlib import contextmanager, nullcontext
from typing import Dict, List
import torch.nn
from .. import skipfiles, variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented, UnspecializeRestartAnalysis, Unsupported
from ..guards import GuardBuilder, install_guard
from ..mutation_guard import GenerationTracker
from ..source import (
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .functions import invoke_and_store_as_constant
from .lists import SliceVariable
from .user_defined import UserDefinedObjectVariable
def generic_call_method_helper(name):
    mod_proxy = tx.output.create_proxy('get_attr', self.module_key, tuple(), {})
    mod_proxy.node.meta['example_value'] = module
    proxy_args, proxy_kwargs = proxy_args_kwargs(args, kwargs)
    from .builder import wrap_fx_proxy
    return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_method', name, args=(mod_proxy, *proxy_args), kwargs=proxy_kwargs))