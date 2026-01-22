from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_valid_format(self):
    validate_format = ['10.0.0.0/24', '1.1.1.1', '1.0.1.1', '255.255.255.255', '6000::/64', '2002:2002::20c:29ff:fe7d:811a', '::1', '2002::', '2002::1']
    for value in validate_format:
        self.assertTrue(self.constraint.validate(value, None))