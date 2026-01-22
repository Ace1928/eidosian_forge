from unittest import mock
from stevedore import enabled
from neutron_lib.tests import _base as base
from neutron_lib.utils import runtime
@mock.patch.object(runtime.driver, 'DriverManager', return_value=_DummyDriver)
@mock.patch.object(runtime, 'LOG')
def test_load_class_by_alias_or_classname_dummy_driver(self, mock_log, mock_driver):
    self.assertEqual(_DummyDriver.driver, runtime.load_class_by_alias_or_classname('ns', 'n'))