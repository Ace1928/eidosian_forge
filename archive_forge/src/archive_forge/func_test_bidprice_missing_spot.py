from decimal import Decimal
from tests.compat import unittest
from boto.emr.instance_group import InstanceGroup
def test_bidprice_missing_spot(self):
    """
        Test InstanceGroup init raises ValueError when market==spot and
        bidprice is not specified.
        """
    with self.assertRaisesRegexp(ValueError, 'bidprice must be specified'):
        InstanceGroup(1, 'MASTER', 'm1.small', 'SPOT', 'master')