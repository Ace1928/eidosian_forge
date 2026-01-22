import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_xml_attribute(self):
    payload = "adminPass='TL0EfN33'"
    expected = "adminPass='***'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "adminPass = 'TL0EfN33'"
    expected = "adminPass = '***'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'adminPass = "TL0EfN33"'
    expected = 'adminPass = "***"'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "admin_pass='TL0EfN33'"
    expected = "admin_pass='***'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "admin_pass = 'TL0EfN33'"
    expected = "admin_pass = '***'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'admin_pass = "TL0EfN33"'
    expected = 'admin_pass = "***"'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "admin_password='TL0EfN33'"
    expected = "admin_password='***'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "admin_password = 'TL0EfN33'"
    expected = "admin_password = '***'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'admin_password = "TL0EfN33"'
    expected = 'admin_password = "***"'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "password='TL0EfN33'"
    expected = "password='***'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = "password = 'TL0EfN33'"
    expected = "password = '***'"
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = 'password = "TL0EfN33"'
    expected = 'password = "***"'
    self.assertEqual(expected, strutils.mask_password(payload))