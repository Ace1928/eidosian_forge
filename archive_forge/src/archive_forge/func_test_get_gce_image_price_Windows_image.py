import sys
import os.path
import unittest
import libcloud.pricing
def test_get_gce_image_price_Windows_image(self):
    cores = 2
    image_name = 'windows-server-2012-r2-dc-core-v20220513'
    size_name = 'n2d-highcpu-2'
    prices = libcloud.pricing.get_pricing('compute', 'gce_images')
    correct_price = float(prices['Windows Server']['any']['price']) * 2
    fetched_price = libcloud.pricing.get_image_price('gce_images', image_name, size_name, cores)
    self.assertTrue(fetched_price == correct_price)