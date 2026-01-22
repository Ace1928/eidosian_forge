from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_invalid_cidr_format(self):
    invalidate_format = ['::/129', 'Invalid cidr', '300.0.0.0/24', '10.0.0.0/33', '10.0.0/24', '10.0/24', '10.0.a.10/24', '8.8.8.0/ 24', '8.8.8.8']
    for cidr in invalidate_format:
        self.assertFalse(self.constraint.validate(cidr, None))