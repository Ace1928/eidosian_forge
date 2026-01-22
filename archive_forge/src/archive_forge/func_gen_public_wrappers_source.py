from __future__ import annotations
import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING
import attrs
import astor
from __future__ import annotations
from ._ki import LOCALS_KEY_KI_PROTECTION_ENABLED
from ._run import GLOBAL_RUN_CONTEXT
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING
from outcome import Outcome
import contextvars
from ._run import _NO_SEND, RunStatistics, Task
from ._entry_queue import TrioToken
from .._abc import Clock
from ._instrumentation import Instrument
from typing import TYPE_CHECKING
from typing import Callable, ContextManager, TYPE_CHECKING
from typing import TYPE_CHECKING, ContextManager
def gen_public_wrappers_source(file: File) -> str:
    """Scan the given .py file for @_public decorators, and generate wrapper
    functions.

    """
    header = [HEADER]
    if file.imports:
        header.append(file.imports)
    if file.platform:
        if 'TYPE_CHECKING' not in file.imports:
            header.append('from typing import TYPE_CHECKING\n')
        if 'import sys' not in file.imports:
            header.append('import sys\n')
        header.append(f'\nassert not TYPE_CHECKING or sys.platform=="{file.platform}"\n')
    generated = [''.join(header)]
    source = astor.code_to_ast.parse_file(file.path)
    method_names = []
    for method in get_public_methods(source):
        assert method.args.args[0].arg == 'self'
        del method.args.args[0]
        method_names.append(method.name)
        for dec in method.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == 'contextmanager':
                is_cm = True
                break
        else:
            is_cm = False
        method.decorator_list = []
        new_args = create_passthrough_args(method)
        if ast.get_docstring(method) is None:
            del method.body[:]
        else:
            del method.body[1:]
        func = astor.to_source(method, indent_with=' ' * 4)
        if is_cm:
            func = func.replace('->Iterator', '->ContextManager')
        template = TEMPLATE.format(' await ' if isinstance(method, ast.AsyncFunctionDef) else ' ', file.modname, method.name + new_args)
        snippet = func + indent(template, ' ' * 4)
        generated.append(snippet)
    method_names.sort()
    generated.insert(1, f'__all__ = {method_names!r}')
    return '\n\n'.join(generated)