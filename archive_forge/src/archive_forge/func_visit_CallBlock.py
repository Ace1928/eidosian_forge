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
def visit_CallBlock(self, node: nodes.CallBlock, frame: Frame) -> None:
    call_frame, macro_ref = self.macro_body(node, frame)
    self.writeline('caller = ')
    self.macro_def(macro_ref, call_frame)
    self.start_write(frame, node)
    self.visit_Call(node.call, frame, forward_caller=True)
    self.end_write(frame)