from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validation_error_special_char(self):
    dns_name = '$openstack.org'
    expected = "'%s' not in valid format. Reason: Name '%s' must be 1-63 characters long, each of which can only be alphanumeric or a hyphen." % (dns_name, dns_name.split('.')[0])
    self.assertFalse(self.constraint.validate(dns_name, self.ctx))
    self.assertEqual(expected, str(self.constraint._error_message))