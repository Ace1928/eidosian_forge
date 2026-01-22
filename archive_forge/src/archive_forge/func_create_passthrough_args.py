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
def create_passthrough_args(funcdef: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Given a function definition, create a string that represents taking all
    the arguments from the function, and passing them through to another
    invocation of the same function.

    Example input: ast.parse("def f(a, *, b): ...")
    Example output: "(a, b=b)"
    """
    call_args = []
    for arg in funcdef.args.args:
        call_args.append(arg.arg)
    if funcdef.args.vararg:
        call_args.append('*' + funcdef.args.vararg.arg)
    for arg in funcdef.args.kwonlyargs:
        call_args.append(arg.arg + '=' + arg.arg)
    if funcdef.args.kwarg:
        call_args.append('**' + funcdef.args.kwarg.arg)
    return '({})'.format(', '.join(call_args))