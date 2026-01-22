from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validation_error_empty_component(self):
    dns_name = '.openstack.org'
    expected = "'%s' not in valid format. Reason: Encountered an empty component." % dns_name
    self.assertFalse(self.constraint.validate(dns_name, self.ctx))
    self.assertEqual(expected, str(self.constraint._error_message))