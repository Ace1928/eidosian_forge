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
def visit_OverlayScope(self, node: nodes.OverlayScope, frame: Frame) -> None:
    ctx = self.temporary_identifier()
    self.writeline(f'{ctx} = {self.derive_context(frame)}')
    self.writeline(f'{ctx}.vars = ')
    self.visit(node.context, frame)
    self.push_context_reference(ctx)
    scope_frame = frame.inner(isolated=True)
    scope_frame.symbols.analyze_node(node)
    self.enter_frame(scope_frame)
    self.blockvisit(node.body, scope_frame)
    self.leave_frame(scope_frame)
    self.pop_context_reference()