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
def visit_Block(self, node: nodes.Block, frame: Frame) -> None:
    """Call a block and register it for the template."""
    level = 0
    if frame.toplevel:
        if self.has_known_extends:
            return
        if self.extends_so_far > 0:
            self.writeline('if parent_template is None:')
            self.indent()
            level += 1
    if node.scoped:
        context = self.derive_context(frame)
    else:
        context = self.get_context_ref()
    if node.required:
        self.writeline(f'if len(context.blocks[{node.name!r}]) <= 1:', node)
        self.indent()
        self.writeline(f'raise TemplateRuntimeError("Required block {node.name!r} not found")', node)
        self.outdent()
    if not self.environment.is_async and frame.buffer is None:
        self.writeline(f'yield from context.blocks[{node.name!r}][0]({context})', node)
    else:
        self.writeline(f'{self.choose_async()}for event in context.blocks[{node.name!r}][0]({context}):', node)
        self.indent()
        self.simple_write('event', frame)
        self.outdent()
    self.outdent(level)