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
def visit_ScopedEvalContextModifier(self, node: nodes.ScopedEvalContextModifier, frame: Frame) -> None:
    old_ctx_name = self.temporary_identifier()
    saved_ctx = frame.eval_ctx.save()
    self.writeline(f'{old_ctx_name} = context.eval_ctx.save()')
    self.visit_EvalContextModifier(node, frame)
    for child in node.body:
        self.visit(child, frame)
    frame.eval_ctx.revert(saved_ctx)
    self.writeline(f'context.eval_ctx.revert({old_ctx_name})')