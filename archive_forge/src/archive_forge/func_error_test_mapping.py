import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def error_test_mapping():
    raise MappingTestError