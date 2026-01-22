import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def legacy_test_mapping():
    return {'foo': 'bar', 'baz': 'quux'}