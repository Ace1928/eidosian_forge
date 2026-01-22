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
def visit_Output(self, node: nodes.Output, frame: Frame) -> None:
    if frame.require_output_check:
        if self.has_known_extends:
            return
        self.writeline('if parent_template is None:')
        self.indent()
    finalize = self._make_finalize()
    body: t.List[t.Union[t.List[t.Any], nodes.Expr]] = []
    for child in node.nodes:
        try:
            if not (finalize.const or isinstance(child, nodes.TemplateData)):
                raise nodes.Impossible()
            const = self._output_child_to_const(child, frame, finalize)
        except (nodes.Impossible, Exception):
            body.append(child)
            continue
        if body and isinstance(body[-1], list):
            body[-1].append(const)
        else:
            body.append([const])
    if frame.buffer is not None:
        if len(body) == 1:
            self.writeline(f'{frame.buffer}.append(')
        else:
            self.writeline(f'{frame.buffer}.extend((')
        self.indent()
    for item in body:
        if isinstance(item, list):
            val = self._output_const_repr(item)
            if frame.buffer is None:
                self.writeline('yield ' + val)
            else:
                self.writeline(val + ',')
        else:
            if frame.buffer is None:
                self.writeline('yield ', item)
            else:
                self.newline(item)
            self._output_child_pre(item, frame, finalize)
            self.visit(item, frame)
            self._output_child_post(item, frame, finalize)
            if frame.buffer is not None:
                self.write(',')
    if frame.buffer is not None:
        self.outdent()
        self.writeline(')' if len(body) == 1 else '))')
    if frame.require_output_check:
        self.outdent()