import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, Set, cast
from docutils import nodes
from docutils.nodes import Element
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util import inspect, typing
def modify_field_list(node: nodes.field_list, annotations: Dict[str, str], suppress_rtype: bool=False) -> None:
    arguments: Dict[str, Dict[str, bool]] = {}
    fields = cast(Iterable[nodes.field], node)
    for field in fields:
        field_name = field[0].astext()
        parts = re.split(' +', field_name)
        if parts[0] == 'param':
            if len(parts) == 2:
                arg = arguments.setdefault(parts[1], {})
                arg['param'] = True
            elif len(parts) > 2:
                name = ' '.join(parts[2:])
                arg = arguments.setdefault(name, {})
                arg['param'] = True
                arg['type'] = True
        elif parts[0] == 'type':
            name = ' '.join(parts[1:])
            arg = arguments.setdefault(name, {})
            arg['type'] = True
        elif parts[0] == 'rtype':
            arguments['return'] = {'type': True}
    for name, annotation in annotations.items():
        if name == 'return':
            continue
        if '*' + name in arguments:
            name = '*' + name
            arguments.get(name)
        elif '**' + name in arguments:
            name = '**' + name
            arguments.get(name)
        else:
            arg = arguments.get(name, {})
        if not arg.get('type'):
            field = nodes.field()
            field += nodes.field_name('', 'type ' + name)
            field += nodes.field_body('', nodes.paragraph('', annotation))
            node += field
        if not arg.get('param'):
            field = nodes.field()
            field += nodes.field_name('', 'param ' + name)
            field += nodes.field_body('', nodes.paragraph('', ''))
            node += field
    if 'return' in annotations and 'return' not in arguments:
        annotation = annotations['return']
        if annotation == 'None' and suppress_rtype:
            return
        field = nodes.field()
        field += nodes.field_name('', 'rtype')
        field += nodes.field_body('', nodes.paragraph('', annotation))
        node += field