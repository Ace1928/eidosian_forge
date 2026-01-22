from unittest import TestCase
from ipykernel.tests import utils
from nbformat.converter import convert
from nbformat.reader import reads
import re
import json
from copy import copy
import unittest
def transform_message(self, msg, expected):
    """
        transform a message into something like the notebook
        """
    SWAP_KEYS = {'output_type': {'pyout': 'execute_result', 'pyerr': 'error'}}
    output = {u'output_type': msg['msg_type']}
    output.update(msg['content'])
    output = self.strip_keys(output)
    for key, swaps in SWAP_KEYS.items():
        if key in output and output[key] in swaps:
            output[key] = swaps[output[key]]
    if 'data' in output and 'data' not in expected:
        output['text'] = output['data']
        del output['data']
    return output