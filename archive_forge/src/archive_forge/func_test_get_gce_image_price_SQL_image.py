import sys
import os.path
import unittest
import libcloud.pricing
def test_get_gce_image_price_SQL_image(self):
    image_name = 'sql-2012-standard-windows-2012-r2-dc-v20220513'
    size_name = 'g1 small'
    prices = libcloud.pricing.get_pricing('compute', 'gce_images')
    correct_price = float(prices['SQL Server']['standard']['price'])
    fetched_price = libcloud.pricing.get_image_price('gce_images', image_name, size_name)
    self.assertTrue(fetched_price == correct_price)