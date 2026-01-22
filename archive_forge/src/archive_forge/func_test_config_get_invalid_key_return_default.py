import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_get_invalid_key_return_default(self):
    from pecan import configuration
    conf = configuration.Config({'a': 1})
    assert conf.get('b', True) is True