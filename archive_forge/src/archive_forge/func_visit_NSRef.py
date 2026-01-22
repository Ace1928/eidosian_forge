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
def visit_NSRef(self, node: nodes.NSRef, frame: Frame) -> None:
    ref = frame.symbols.ref(node.name)
    self.writeline(f'if not isinstance({ref}, Namespace):')
    self.indent()
    self.writeline('raise TemplateRuntimeError("cannot assign attribute on non-namespace object")')
    self.outdent()
    self.writeline(f'{ref}[{node.attr!r}]')