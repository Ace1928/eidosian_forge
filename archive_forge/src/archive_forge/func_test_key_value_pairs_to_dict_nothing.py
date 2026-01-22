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
def test_key_value_pairs_to_dict_nothing(self):
    self.assertEqual({}, utils.key_value_pairs_to_dict(None))
    self.assertEqual({}, utils.key_value_pairs_to_dict([]))