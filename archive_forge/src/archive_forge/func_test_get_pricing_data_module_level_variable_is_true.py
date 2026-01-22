import sys
import os.path
import unittest
import libcloud.pricing
def test_get_pricing_data_module_level_variable_is_true(self):
    self.assertEqual(libcloud.pricing.PRICING_DATA['compute'], {})
    self.assertEqual(libcloud.pricing.PRICING_DATA['storage'], {})
    libcloud.pricing.CACHE_ALL_PRICING_DATA = True
    pricing = libcloud.pricing.get_pricing(driver_type='compute', driver_name='foo', pricing_file_path=PRICING_FILE_PATH, cache_all=False)
    self.assertEqual(pricing['1'], 1.0)
    self.assertEqual(pricing['2'], 2.0)
    self.assertEqual(len(libcloud.pricing.PRICING_DATA['compute']), 3)
    self.assertTrue('foo' in libcloud.pricing.PRICING_DATA['compute'])
    self.assertTrue('bar' in libcloud.pricing.PRICING_DATA['compute'])
    self.assertTrue('baz' in libcloud.pricing.PRICING_DATA['compute'])