import sys
import os.path
import unittest
import libcloud.pricing
def test_get_pricing_data_caching(self):
    self.assertEqual(libcloud.pricing.PRICING_DATA['compute'], {})
    self.assertEqual(libcloud.pricing.PRICING_DATA['storage'], {})
    pricing = libcloud.pricing.get_pricing(driver_type='compute', driver_name='foo', pricing_file_path=PRICING_FILE_PATH)
    self.assertEqual(pricing['1'], 1.0)
    self.assertEqual(pricing['2'], 2.0)
    self.assertEqual(len(libcloud.pricing.PRICING_DATA['compute']), 1)
    self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
    pricing = libcloud.pricing.get_pricing(driver_type='compute', driver_name='baz', pricing_file_path=PRICING_FILE_PATH)
    self.assertEqual(pricing['1'], 5.0)
    self.assertEqual(pricing['2'], 6.0)
    self.assertEqual(len(libcloud.pricing.PRICING_DATA['compute']), 2)
    self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
    self.assertTrue('baz' in libcloud.pricing.PRICING_DATA['compute'])