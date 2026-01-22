import warnings
from ..helpers import quote_string, random_string, stringify_param_value
from .commands import AsyncGraphCommands, GraphCommands
from .edge import Edge  # noqa
from .node import Node  # noqa
from .path import Path  # noqa
def call_procedure(self, procedure, *args, read_only=False, **kwagrs):
    args = [quote_string(arg) for arg in args]
    q = f'CALL {procedure}({','.join(args)})'
    y = kwagrs.get('y', None)
    if y is not None:
        q += f'YIELD {','.join(y)}'
    return self.query(q, read_only=read_only)