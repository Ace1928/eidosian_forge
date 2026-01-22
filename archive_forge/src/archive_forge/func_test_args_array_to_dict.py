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
def test_args_array_to_dict(self):
    my_args = {'matching_metadata': ['str=foo', 'int=1', 'bool=true', 'list=[1, 2, 3]', 'dict={"foo": "bar"}'], 'other': 'value'}
    cleaned_dict = utils.args_array_to_dict(my_args, 'matching_metadata')
    self.assertEqual({'matching_metadata': {'str': 'foo', 'int': 1, 'bool': True, 'list': [1, 2, 3], 'dict': {'foo': 'bar'}}, 'other': 'value'}, cleaned_dict)