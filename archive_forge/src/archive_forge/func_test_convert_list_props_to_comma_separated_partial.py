import builtins
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
def test_convert_list_props_to_comma_separated_partial(self):
    data = {'prop1': [1, 2, 3], 'prop2': [1, 2.5, 'item3']}
    result = utils.convert_list_props_to_comma_separated(data, props=['prop2'])
    self.assertEqual([1, 2, 3], result['prop1'])
    self.assertEqual('1, 2.5, item3', result['prop2'])