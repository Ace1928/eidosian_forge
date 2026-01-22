from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
import subprocess
from unittest import mock
import six
from gslib import context_config
from gslib import exception
from gslib.tests import testcase
from gslib.tests.testcase import base
from gslib.tests.util import SetBotoConfigForTest
def test_context_config_is_a_singleton(self):
    first = context_config.create_context_config(self.mock_logger)
    with self.assertRaises(context_config.ContextConfigSingletonAlreadyExistsError):
        context_config.create_context_config(self.mock_logger)
    second = context_config.get_context_config()
    self.assertEqual(first, second)