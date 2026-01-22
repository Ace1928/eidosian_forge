import os
from traitlets import Any, Unicode, List, Dict, Union, Instance
from ipywidgets import DOMWidget
from ipywidgets.widgets.widget import widget_serialization
from .Template import Template, get_template
from ._version import semver
from .ForceLoad import force_load_instance
import inspect
from importlib import import_module
def to_ref_structure(obj, path):
    if isinstance(obj, list):
        return [to_ref_structure(item, [*path, index]) for index, item in enumerate(obj)]
    if isinstance(obj, dict):
        return {k: to_ref_structure(v, [*path, k]) for k, v in obj.items()}
    return {OBJECT_REF: name, 'path': path, 'id': id(obj)}