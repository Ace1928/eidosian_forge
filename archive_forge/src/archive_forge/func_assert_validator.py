import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def assert_validator(tester, attribute, attribute_dict, keyword, value):
    tester.assertIn(list(value)[0], validators.validators, '%s is not a known validator for %s.' % (value, attribute))