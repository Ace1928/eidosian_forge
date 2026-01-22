import enum
import functools
import inspect
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
def zip_arguments(schema: torch.FunctionSchema, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Iterator[Tuple[torch.Argument, Any]]:
    schema_args = schema.arguments[:len(args)]
    schema_kwargs = {arg.name: arg for arg in schema.arguments[len(args):]}
    yield from zip(schema_args, args)
    for _, argument, value in zip_by_key(schema_kwargs, kwargs):
        yield (argument, value)