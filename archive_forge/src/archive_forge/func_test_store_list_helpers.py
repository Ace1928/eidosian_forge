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
def test_store_list_helpers(self):
    self.actions.add_importing_stores(['foo', 'bar', 'baz'])
    self.actions.remove_importing_stores(['bar'])
    self.actions.add_failed_stores(['foo', 'bar'])
    self.actions.remove_failed_stores(['foo'])
    self.assertEqual({'speed': '88mph', 'os_glance_importing_to_stores': 'baz,foo', 'os_glance_failed_import': 'bar'}, self.image.extra_properties)