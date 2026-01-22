from unittest import mock
import novaclient
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient import utils as nutils
from novaclient.v2 import versions
def test_several_methods_with_same_name_in_one_module(self):

    class A(object):
        api_version = api_versions.APIVersion('777.777')

        @api_versions.wraps('777.777')
        def f(self):
            return 1

    class B(object):
        api_version = api_versions.APIVersion('777.777')

        @api_versions.wraps('777.777')
        def f(self):
            return 2
    self.assertEqual(1, A().f())
    self.assertEqual(2, B().f())