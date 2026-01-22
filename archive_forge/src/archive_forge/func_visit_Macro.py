import typing as t
from contextlib import contextmanager
from functools import update_wrapper
from io import StringIO
from itertools import chain
from keyword import iskeyword as is_python_keyword
from markupsafe import escape
from markupsafe import Markup
from . import nodes
from .exceptions import TemplateAssertionError
from .idtracking import Symbols
from .idtracking import VAR_LOAD_ALIAS
from .idtracking import VAR_LOAD_PARAMETER
from .idtracking import VAR_LOAD_RESOLVE
from .idtracking import VAR_LOAD_UNDEFINED
from .nodes import EvalContext
from .optimizer import Optimizer
from .utils import _PassArg
from .utils import concat
from .visitor import NodeVisitor
def visit_Macro(self, node: nodes.Macro, frame: Frame) -> None:
    macro_frame, macro_ref = self.macro_body(node, frame)
    self.newline()
    if frame.toplevel:
        if not node.name.startswith('_'):
            self.write(f'context.exported_vars.add({node.name!r})')
        self.writeline(f'context.vars[{node.name!r}] = ')
    self.write(f'{frame.symbols.ref(node.name)} = ')
    self.macro_def(macro_ref, macro_frame)