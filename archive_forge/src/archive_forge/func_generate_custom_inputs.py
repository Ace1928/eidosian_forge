import os
import sys
import simplejson as json
from ..scripts.instance import import_module
def generate_custom_inputs(desc_inputs):
    """
    Generates a bunch of custom input dictionaries in order to generate
    as many outputs as possible (to get their path templates).
    Currently only works with flag inputs and inputs with defined value
    choices.
    """
    custom_input_dicts = []
    for desc_input in desc_inputs:
        if desc_input['type'] == 'Flag':
            custom_input_dicts.append({desc_input['id']: True})
        elif desc_input.get('value-choices') and (not desc_input.get('list')):
            for value in desc_input['value-choices']:
                custom_input_dicts.append({desc_input['id']: value})
    return custom_input_dicts