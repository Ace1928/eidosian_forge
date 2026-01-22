import collections
import json
import os
from ._py_components_generation import (
from .base_component import ComponentRegistry
def load_components(metadata_path, namespace='default_namespace'):
    """Load React component metadata into a format Dash can parse.

    Usage: load_components('../../component-suites/lib/metadata.json')

    Keyword arguments:
    metadata_path -- a path to a JSON file created by
    [`react-docgen`](https://github.com/reactjs/react-docgen).

    Returns:
    components -- a list of component objects with keys
    `type`, `valid_kwargs`, and `setup`.
    """
    ComponentRegistry.registry.add(namespace)
    components = []
    data = _get_metadata(metadata_path)
    for componentPath in data:
        componentData = data[componentPath]
        name = componentPath.split('/').pop().split('.')[0]
        component = generate_class(name, componentData['props'], componentData['description'], namespace, None)
        components.append(component)
    return components