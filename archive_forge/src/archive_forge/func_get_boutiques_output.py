import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def get_boutiques_output(outputs, name, spec, interface, tool_inputs):
    """
    Returns a dictionary containing the Boutiques output corresponding
    to a Nipype output.

    Args:
      * outputs: outputs of the Nipype interface.
      * name: name of the Nipype output.
      * spec: Nipype output spec.
      * interface: Nipype interface.
      * tool_inputs: list of tool inputs (as produced by method
        get_boutiques_input).

    Assumes that:
      * Output names are unique.
      * Input values involved in the path template are defined.
      * Output files are written in the current directory.
      * There is a single output value (output lists are not supported).
    """
    output = {}
    output['name'] = name.replace('_', ' ').capitalize()
    unique_id = True
    for inp in tool_inputs:
        if inp['id'] == name:
            unique_id = False
            break
    output['id'] = name if unique_id else name + '_outfile'
    output['path-template'] = ''
    output['optional'] = True
    output['description'] = get_description_from_spec(outputs, name, spec)
    try:
        output_value = interface._list_outputs()[name]
    except TypeError:
        output_value = None
    except AttributeError:
        output_value = None
    except KeyError:
        output_value = None
    if isinstance(output_value, list) or type(spec.handler).__name__ == 'OutputMultiObject' or type(spec.handler).__name__ == 'List':
        output['list'] = True
        if output_value:
            extensions = []
            for val in output_value:
                extensions.append(os.path.splitext(val)[1])
            if len(set(extensions)) == 1:
                output['path-template'] = '*' + extensions[0]
            else:
                output['path-template'] = '*'
            return output
    if output_value:
        output['path-template'] = os.path.relpath(output_value)
    else:
        output['path-template'] = ''
    return output