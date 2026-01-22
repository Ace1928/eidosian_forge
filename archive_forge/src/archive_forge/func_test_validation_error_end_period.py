from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validation_error_end_period(self):
    dns_name = 'myvm.openstack.'
    expected = "'%s' is a FQDN. It should be a relative domain name." % dns_name
    self.assertFalse(self.constraint.validate(dns_name, self.ctx))
    self.assertEqual(expected, str(self.constraint._error_message))