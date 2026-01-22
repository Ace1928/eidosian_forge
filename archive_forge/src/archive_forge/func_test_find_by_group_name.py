import collections
import io
import sys
from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient.tests.unit import utils as test_utils
from cinderclient import utils
def test_find_by_group_name(self):
    display_manager = FakeDisplayManager(None)
    output = utils.find_resource(display_manager, 'entity_three', is_group=True, list_volume=True)
    self.assertEqual(display_manager.get('4242', list_volume=True), output)