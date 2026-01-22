import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def get_description_from_spec(obj, name, spec):
    """
    Generates a description based on the input or output spec.
    """
    if not spec.desc:
        spec.desc = 'No description provided.'
    spec_info = spec.full_info(obj, name, None)
    boutiques_description = (spec_info.capitalize() + '. ' + spec.desc.capitalize()).replace('\n', '')
    if not boutiques_description.endswith('.'):
        boutiques_description += '.'
    return boutiques_description