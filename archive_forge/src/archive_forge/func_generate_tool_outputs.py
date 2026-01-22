import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def generate_tool_outputs(outputs, interface, tool_desc, verbose, first_run):
    for name, spec in sorted(outputs.traits(transient=None).items()):
        output = get_boutiques_output(outputs, name, spec, interface, tool_desc['inputs'])
        if first_run:
            tool_desc['output-files'].append(output)
            if output.get('value-key'):
                tool_desc['command-line'] += output['value-key'] + ' '
            if verbose:
                print('-> Adding output ' + output['name'])
        else:
            for existing_output in tool_desc['output-files']:
                if output['id'] == existing_output['id'] and existing_output['path-template'] == '':
                    existing_output['path-template'] = output['path-template']
                    break
            if output.get('value-key') and output['value-key'] not in tool_desc['command-line']:
                tool_desc['command-line'] += output['value-key'] + ' '
    if len(tool_desc['output-files']) == 0:
        raise Exception('Tool has no output.')