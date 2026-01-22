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
@optimizeconst
def visit_Filter(self, node: nodes.Filter, frame: Frame) -> None:
    with self._filter_test_common(node, frame, True):
        if node.node is not None:
            self.visit(node.node, frame)
        elif frame.eval_ctx.volatile:
            self.write(f'(Markup(concat({frame.buffer})) if context.eval_ctx.autoescape else concat({frame.buffer}))')
        elif frame.eval_ctx.autoescape:
            self.write(f'Markup(concat({frame.buffer}))')
        else:
            self.write(f'concat({frame.buffer})')