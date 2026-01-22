import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def get_boutiques_input(inputs, interface, input_name, spec, verbose, handler=None, input_number=None):
    """
    Returns a dictionary containing the Boutiques input corresponding
    to a Nipype input.

    Args:
      * inputs: inputs of the Nipype interface.
      * interface: Nipype interface.
      * input_name: name of the Nipype input.
      * spec: Nipype input spec.
      * verbose: print information messages.
      * handler: used when handling compound inputs, which don't have their
        own input spec
      * input_number: used when handling compound inputs to assign each a
        unique ID

    Assumes that:
      * Input names are unique.
    """
    inp = {}
    if input_number:
        inp['id'] = input_name + '_' + str(input_number + 1)
    else:
        inp['id'] = input_name
    inp['name'] = input_name.replace('_', ' ').capitalize()
    if handler is None:
        trait_handler = spec.handler
    else:
        trait_handler = handler
    handler_type = type(trait_handler).__name__
    if handler_type == 'TraitCompound':
        input_list = []
        for i in range(0, len(trait_handler.handlers)):
            inp = get_boutiques_input(inputs, interface, input_name, spec, verbose, trait_handler.handlers[i], i)
            inp['optional'] = True
            input_list.append(inp)
        return input_list
    if handler_type == 'File' or handler_type == 'Directory':
        inp['type'] = 'File'
    elif handler_type == 'Int':
        inp['type'] = 'Number'
        inp['integer'] = True
    elif handler_type == 'Float':
        inp['type'] = 'Number'
    elif handler_type == 'Bool':
        inp['type'] = 'Flag'
    else:
        inp['type'] = 'String'
    if handler_type == 'Range':
        inp['type'] = 'Number'
        if trait_handler._low is not None:
            inp['minimum'] = trait_handler._low
        if trait_handler._high is not None:
            inp['maximum'] = trait_handler._high
        if trait_handler._exclude_low:
            inp['exclusive-minimum'] = True
        if trait_handler._exclude_high:
            inp['exclusive-maximum'] = True
    if handler_type == 'List':
        inp['list'] = True
        item_type = trait_handler.item_trait.trait_type
        item_type_name = type(item_type).__name__
        if item_type_name == 'Int':
            inp['integer'] = True
            inp['type'] = 'Number'
        elif item_type_name == 'Float':
            inp['type'] = 'Number'
        elif item_type_name == 'File':
            inp['type'] = 'File'
        elif item_type_name == 'Enum':
            value_choices = item_type.values
            if value_choices is not None:
                if all((isinstance(n, int) for n in value_choices)):
                    inp['type'] = 'Number'
                    inp['integer'] = True
                elif all((isinstance(n, float) for n in value_choices)):
                    inp['type'] = 'Number'
                inp['value-choices'] = value_choices
        else:
            inp['type'] = 'String'
        if trait_handler.minlen != 0:
            inp['min-list-entries'] = trait_handler.minlen
        if trait_handler.maxlen != sys.maxsize:
            inp['max-list-entries'] = trait_handler.maxlen
        if spec.sep:
            inp['list-separator'] = spec.sep
    if handler_type == 'Tuple':
        inp['list'] = True
        inp['min-list-entries'] = len(spec.default)
        inp['max-list-entries'] = len(spec.default)
        input_type = type(spec.default[0]).__name__
        if input_type == 'int':
            inp['type'] = 'Number'
            inp['integer'] = True
        elif input_type == 'float':
            inp['type'] = 'Number'
        else:
            inp['type'] = 'String'
    if handler_type == 'InputMultiObject':
        inp['type'] = 'File'
        inp['list'] = True
        if spec.sep:
            inp['list-separator'] = spec.sep
    inp['value-key'] = '[' + input_name.upper() + ']'
    flag, flag_sep = get_command_line_flag(spec, inp['type'] == 'Flag', input_name)
    if flag is not None:
        inp['command-line-flag'] = flag
    if flag_sep is not None:
        inp['command-line-flag-separator'] = flag_sep
    inp['description'] = get_description_from_spec(inputs, input_name, spec)
    if not (hasattr(spec, 'mandatory') and spec.mandatory):
        inp['optional'] = True
    else:
        inp['optional'] = False
    if spec.usedefault:
        inp['default-value'] = spec.default_value()[1]
    if spec.requires is not None:
        inp['requires-inputs'] = spec.requires
    try:
        value_choices = trait_handler.values
    except AttributeError:
        pass
    else:
        if value_choices is not None:
            if all((isinstance(n, int) for n in value_choices)):
                inp['type'] = 'Number'
                inp['integer'] = True
            elif all((isinstance(n, float) for n in value_choices)):
                inp['type'] = 'Number'
            inp['value-choices'] = value_choices
    return inp