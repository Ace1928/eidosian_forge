import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def fill_in_missing_output_path(output, output_name, tool_inputs):
    """
    Creates a path template for outputs that are missing one
    This is needed for the descriptor to be valid (path template is required)
    """
    found = False
    for input in tool_inputs:
        if input['name'] == output_name:
            output['path-template'] = input['value-key']
            found = True
            break
    if not found:
        output['path-template'] = output['id']
    return output