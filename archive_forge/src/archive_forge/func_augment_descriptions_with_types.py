import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, Set, cast
from docutils import nodes
from docutils.nodes import Element
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util import inspect, typing
def augment_descriptions_with_types(node: nodes.field_list, annotations: Dict[str, str], force_rtype: bool) -> None:
    fields = cast(Iterable[nodes.field], node)
    has_description = set()
    has_type = set()
    for field in fields:
        field_name = field[0].astext()
        parts = re.split(' +', field_name)
        if parts[0] == 'param':
            if len(parts) == 2:
                has_description.add(parts[1])
            elif len(parts) > 2:
                name = ' '.join(parts[2:])
                has_description.add(name)
                has_type.add(name)
        elif parts[0] == 'type':
            name = ' '.join(parts[1:])
            has_type.add(name)
        elif parts[0] in ('return', 'returns'):
            has_description.add('return')
        elif parts[0] == 'rtype':
            has_type.add('return')
    for name, annotation in annotations.items():
        if name in ('return', 'returns'):
            continue
        if '*' + name in has_description:
            name = '*' + name
        elif '**' + name in has_description:
            name = '**' + name
        if name in has_description and name not in has_type:
            field = nodes.field()
            field += nodes.field_name('', 'type ' + name)
            field += nodes.field_body('', nodes.paragraph('', annotation))
            node += field
    if 'return' in annotations:
        rtype = annotations['return']
        if 'return' not in has_type and ('return' in has_description or (force_rtype and rtype != 'None')):
            field = nodes.field()
            field += nodes.field_name('', 'rtype')
            field += nodes.field_body('', nodes.paragraph('', rtype))
            node += field