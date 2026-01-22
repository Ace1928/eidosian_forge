import sys
from unittest import mock
from libcloud.test import unittest
from libcloud.compute.drivers.libvirt_driver import LibvirtNodeDriver, have_libvirt
def test_ip_map(self, *args, **keywargs):
    driver = LibvirtNodeDriver('')
    arp_table = driver._parse_ip_table_neigh(self.ip_output_str)
    self._assert_arp_table(arp_table)