import ast
import dataclasses
import inspect
import os
from functools import partial
from typing import Callable, Dict, List
from torch._jit_internal import FAKE_FILENAME_PREFIX, is_optional
from torch._sources import ParsedDef, SourceContext
def synthesize__init__(cls) -> ParsedDef:
    if any((field.default_factory is not dataclasses.MISSING for field in dataclasses.fields(cls))):
        raise NotImplementedError('Default factory initializers are not supported in TorchScript dataclasses')
    signature = inspect.signature(cls.__init__)
    init_vars: List[str] = []
    params = []
    for name, param in signature.parameters.items():
        ann = param.annotation
        if isinstance(ann, dataclasses.InitVar):
            init_vars.append(name)
            params.append(param.replace(annotation=ann.type))
        else:
            params.append(param)
    signature = signature.replace(parameters=params)
    body = [f'self.{field.name} = {field.name}' for field in dataclasses.fields(cls) if field.init and field.name not in init_vars]
    if hasattr(cls, '__post_init__'):
        body.append('self.__post_init__(' + ', '.join(init_vars) + ')')
    return compose_fn(cls, '__init__', body or ['pass'], signature=str(signature))