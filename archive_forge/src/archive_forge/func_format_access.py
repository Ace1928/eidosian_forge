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
def format_access(access: Access):
    message.write(f'{access.operator}\n{access.type}')
    if access.aliases:
        message.write(' argument(s) ' + ', '.join(access.aliases))
        if access.is_output:
            message.write(', and to')
    if access.is_output:
        message.write(' the output')
    message.write(f'\nWith stack trace:\n{''.join(access.stack_trace.format())}\n')