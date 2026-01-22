from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_invalidate_ipv4_format(self):
    invalidate_format = [None, 123, '1.1', '1.1.', '1.1.1', '1.1.1.', '1.1.1.256', 'invalidate format', '1.a.1.1']
    for ip in invalidate_format:
        self.assertFalse(self.constraint.validate(ip, None))