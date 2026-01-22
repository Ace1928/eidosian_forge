import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def invalid_type_test_mapping():
    return 'foo'