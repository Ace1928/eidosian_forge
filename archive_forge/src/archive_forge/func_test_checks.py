from unittest import mock
from oslo_upgradecheck.upgradecheck import Code
from heat.cmd import status
from heat.tests import common
@mock.patch('oslo_utils.fileutils.is_json')
def test_checks(self, mock_util):
    mock_util.return_value = False
    for name, func in self.cmd._upgrade_checks:
        if isinstance(func, tuple):
            func_name, kwargs = func
            result = func_name(self, **kwargs)
        else:
            result = func(self)
        self.assertEqual(Code.SUCCESS, result.code)