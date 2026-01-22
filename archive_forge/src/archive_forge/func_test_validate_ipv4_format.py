from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validate_ipv4_format(self):
    validate_format = ['1.1.1.1', '1.0.1.1', '255.255.255.255']
    for ip in validate_format:
        self.assertTrue(self.constraint.validate(ip, None))