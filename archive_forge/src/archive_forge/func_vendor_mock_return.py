import copy
from unittest import mock
import fixtures
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
def vendor_mock_return():
    return ('file.yaml', copy.deepcopy(test_shell.PUBLIC_1))