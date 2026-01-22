import copy
import os
import shutil
import warnings
import sys
import importlib
import uuid
import hashlib
from ._all_keywords import julia_keywords
from ._py_components_generation import reorder_props
def get_jl_prop_types(type_object):
    """Mapping from the PropTypes js type object to the Julia type."""

    def shape_or_exact():
        return 'lists containing elements {}.\n{}'.format(', '.join(("'{}'".format(t) for t in type_object['value'])), 'Those elements have the following types:\n{}'.format('\n'.join((create_prop_docstring_jl(prop_name=prop_name, type_object=prop, required=prop['required'], description=prop.get('description', ''), indent_num=1) for prop_name, prop in type_object['value'].items()))))
    return dict(array=lambda: 'Array', bool=lambda: 'Bool', number=lambda: 'Real', string=lambda: 'String', object=lambda: 'Dict', any=lambda: 'Bool | Real | String | Dict | Array', element=lambda: 'dash component', node=lambda: 'a list of or a singular dash component, string or number', enum=lambda: 'a value equal to: {}'.format(', '.join(('{}'.format(str(t['value'])) for t in type_object['value']))), union=lambda: '{}'.format(' | '.join(('{}'.format(get_jl_type(subType)) for subType in type_object['value'] if get_jl_type(subType) != ''))), arrayOf=lambda: 'Array' + (' of {}s'.format(get_jl_type(type_object['value'])) if get_jl_type(type_object['value']) != '' else ''), objectOf=lambda: 'Dict with Strings as keys and values of type {}'.format(get_jl_type(type_object['value'])), shape=shape_or_exact, exact=shape_or_exact)