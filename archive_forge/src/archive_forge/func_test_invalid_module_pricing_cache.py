import sys
import os.path
import unittest
import libcloud.pricing
def test_invalid_module_pricing_cache(self):
    libcloud.pricing.PRICING_DATA['compute']['foo'] = {1: 1}
    self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
    libcloud.pricing.invalidate_module_pricing_cache(driver_type='compute', driver_name='foo')
    self.assertFalse('foo' in libcloud.pricing.PRICING_DATA['compute'])
    libcloud.pricing.invalidate_module_pricing_cache(driver_type='compute', driver_name='foo1')