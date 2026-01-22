from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_parser, '_parse_check', base.FakeCheck)
def test_oneele_multi(self):
    result = _parser._parse_list_rule([['rule1', 'rule2']])
    self.assertIsInstance(result, _checks.AndCheck)
    self.assertEqual(2, len(result.rules))
    for i, value in enumerate(['rule1', 'rule2']):
        self.assertIsInstance(result.rules[i], base.FakeCheck)
        self.assertEqual(value, result.rules[i].result)
    self.assertEqual('(rule1 and rule2)', str(result))