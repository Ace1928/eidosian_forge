import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def get_boutiques_output_from_inp(inputs, inp_spec, inp_name):
    """
    Takes a Nipype input representing an output file and generates a
    Boutiques output for it
    """
    output = {}
    output['name'] = inp_name.replace('_', ' ').capitalize()
    output['id'] = inp_name
    output['optional'] = True
    output['description'] = get_description_from_spec(inputs, inp_name, inp_spec)
    if not (hasattr(inp_spec, 'mandatory') and inp_spec.mandatory):
        output['optional'] = True
    else:
        output['optional'] = False
    if inp_spec.usedefault:
        output['default-value'] = inp_spec.default_value()[1]
    if isinstance(inp_spec.name_source, list):
        source = inp_spec.name_source[0]
    else:
        source = inp_spec.name_source
    output['path-template'] = inp_spec.name_template.replace('%s', '[' + source.upper() + ']')
    output['value-key'] = '[' + inp_name.upper() + ']'
    flag, flag_sep = get_command_line_flag(inp_spec)
    if flag is not None:
        output['command-line-flag'] = flag
    if flag_sep is not None:
        output['command-line-flag-separator'] = flag_sep
    return output