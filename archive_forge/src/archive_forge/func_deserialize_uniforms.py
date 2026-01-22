import numbers
import math
from ipywidgets import Widget, widget_serialization
def deserialize_uniforms(serialized, obj):
    """Deserialize a uniform dict"""
    uniforms = {}
    for name, uniform in serialized.items():
        t = uniform['type']
        value = uniform['value']
        if t == 't':
            uniforms[name] = {'value': widget_serialization['from_json'](value, None)}
        else:
            uniforms[name].value = uniform.value
    return uniforms