import json
from pydeck.types.base import PydeckType
def default_serialize(o, remap_function=lower_camel_case_keys):
    """Default method for rendering JSON from a dictionary"""
    if issubclass(type(o), PydeckType):
        return repr(o)
    attrs = vars(o)
    attrs = {k: v for k, v in attrs.items() if v is not None}
    for ignore_attr in IGNORE_KEYS:
        if attrs.get(ignore_attr):
            del attrs[ignore_attr]
    if remap_function:
        remap_function(attrs)
    return attrs