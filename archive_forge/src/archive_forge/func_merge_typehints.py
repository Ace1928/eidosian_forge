import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, Set, cast
from docutils import nodes
from docutils.nodes import Element
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util import inspect, typing
def merge_typehints(app: Sphinx, domain: str, objtype: str, contentnode: Element) -> None:
    if domain != 'py':
        return
    if app.config.autodoc_typehints not in ('both', 'description'):
        return
    try:
        signature = cast(addnodes.desc_signature, contentnode.parent[0])
        if signature['module']:
            fullname = '.'.join([signature['module'], signature['fullname']])
        else:
            fullname = signature['fullname']
    except KeyError:
        return
    annotations = app.env.temp_data.get('annotations', {})
    if annotations.get(fullname, {}):
        field_lists = [n for n in contentnode if isinstance(n, nodes.field_list)]
        if field_lists == []:
            field_list = insert_field_list(contentnode)
            field_lists.append(field_list)
        for field_list in field_lists:
            if app.config.autodoc_typehints_description_target == 'all':
                if objtype == 'class':
                    modify_field_list(field_list, annotations[fullname], suppress_rtype=True)
                else:
                    modify_field_list(field_list, annotations[fullname])
            elif app.config.autodoc_typehints_description_target == 'documented_params':
                augment_descriptions_with_types(field_list, annotations[fullname], force_rtype=True)
            else:
                augment_descriptions_with_types(field_list, annotations[fullname], force_rtype=False)