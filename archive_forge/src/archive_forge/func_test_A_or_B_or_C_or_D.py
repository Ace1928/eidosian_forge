from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
def test_A_or_B_or_C_or_D(self):
    result = _parser._parse_text_rule('@ or ! or @ or !')
    self.assertEqual('(@ or ! or @ or !)', str(result))