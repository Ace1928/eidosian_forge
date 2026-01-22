from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validate_ipv6_format(self):
    validate_format = ['2002:2002::20c:29ff:fe7d:811a', '::1', '2002::', '2002::1']
    for ip in validate_format:
        self.assertTrue(self.constraint.validate(ip, None))