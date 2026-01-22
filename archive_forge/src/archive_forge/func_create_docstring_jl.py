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
def create_docstring_jl(component_name, props, description):
    """Create the Dash component docstring.
    Parameters
    ----------
    component_name: str
        Component name
    props: dict
        Dictionary with {propName: propMetadata} structure
    description: str
        Component description
    Returns
    -------
    str
        Dash component docstring
    """
    props = reorder_props(props=props)
    return 'A{n} {name} component.\n{description}\nKeyword arguments:\n{args}'.format(n='n' if component_name[0].lower() in 'aeiou' else '', name=component_name, description=description, args='\n'.join((create_prop_docstring_jl(prop_name=p, type_object=prop['type'] if 'type' in prop else prop['flowType'], required=prop['required'], description=prop['description'], indent_num=0) for p, prop in filter_props(props).items())))