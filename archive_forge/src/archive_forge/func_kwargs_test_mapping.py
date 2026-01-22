import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def kwargs_test_mapping(**kwargs):
    return kwargs