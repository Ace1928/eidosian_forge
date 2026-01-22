from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def reorder_props(props):
    """If "children" is in props, then move it to the front to respect dash
    convention, then 'id', then the remaining props sorted by prop name
    Parameters
    ----------
    props: dict
        Dictionary with {propName: propMetadata} structure
    Returns
    -------
    dict
        Dictionary with {propName: propMetadata} structure
    """
    props1 = [('children', '')] if 'children' in props else []
    props2 = [('id', '')] if 'id' in props else []
    return OrderedDict(props1 + props2 + sorted(list(props.items())))