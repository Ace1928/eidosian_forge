from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def generate_imports(project_shortname, components):
    with open(os.path.join(project_shortname, '_imports_.py'), 'w', encoding='utf-8') as f:
        component_imports = '\n'.join((f'from .{x} import {x}' for x in components))
        all_list = ',\n'.join((f'    "{x}"' for x in components))
        imports_string = f'{component_imports}\n\n__all__ = [\n{all_list}\n]'
        f.write(imports_string)