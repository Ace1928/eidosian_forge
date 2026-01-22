import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
def test_remove_location_for_store_last_location(self):
    self.image.locations = [{'metadata': {'store': 'foo'}}]
    self.actions.remove_location_for_store('foo')
    self.assertEqual([], self.image.locations)
    self.assertIsNone(self.image.checksum)
    self.assertIsNone(self.image.os_hash_algo)
    self.assertIsNone(self.image.os_hash_value)
    self.assertIsNone(self.image.size)