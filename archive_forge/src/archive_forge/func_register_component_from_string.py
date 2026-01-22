import os
from traitlets import Unicode
from ipywidgets import DOMWidget
from ._version import semver
def register_component_from_string(name, value):
    components = vue_component_registry
    if name in components.keys():
        comp = components[name]
        comp.component = value
    else:
        comp = VueComponent(name=name, component=value)
        components[name] = comp