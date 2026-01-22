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
def visit_AssignBlock(self, node: nodes.AssignBlock, frame: Frame) -> None:
    self.push_assign_tracking()
    block_frame = frame.inner()
    block_frame.require_output_check = False
    block_frame.symbols.analyze_node(node)
    self.enter_frame(block_frame)
    self.buffer(block_frame)
    self.blockvisit(node.body, block_frame)
    self.newline(node)
    self.visit(node.target, frame)
    self.write(' = (Markup if context.eval_ctx.autoescape else identity)(')
    if node.filter is not None:
        self.visit_Filter(node.filter, block_frame)
    else:
        self.write(f'concat({block_frame.buffer})')
    self.write(')')
    self.pop_assign_tracking(frame)
    self.leave_frame(block_frame)