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
def test_could_be_json(self):
    self.assertIsNone(utils.get_json_data(b'{"hahaha, just kidding\x00'))