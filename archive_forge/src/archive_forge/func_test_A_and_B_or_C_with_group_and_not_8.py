from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
def test_A_and_B_or_C_with_group_and_not_8(self):
    result = _parser._parse_text_rule('@ and ( ! or not @ )')
    self.assertEqual('(@ and (! or not @))', str(result))