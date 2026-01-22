from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validation_error_tld_allnumeric(self):
    dns_name = 'openstack.123.'
    expected = "'%s' not in valid format. Reason: TLD '%s' must not be all numeric." % (dns_name, dns_name.split('.')[1])
    self.assertFalse(self.constraint.validate(dns_name, self.ctx))
    self.assertEqual(expected, str(self.constraint._error_message))