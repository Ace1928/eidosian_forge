import sys
import os.path
import unittest
import libcloud.pricing
def test_get_gce_image_price_RHEL_image(self):
    cores = 2
    image_name = 'rhel-7-v20220519'
    size_name = 'n2d-highcpu-2'
    prices = libcloud.pricing.get_pricing('compute', 'gce_images')
    correct_price = float(prices['RHEL']['4vcpu or less']['price'])
    fetched_price = libcloud.pricing.get_image_price('gce_images', image_name, size_name, cores)
    self.assertTrue(fetched_price == correct_price)