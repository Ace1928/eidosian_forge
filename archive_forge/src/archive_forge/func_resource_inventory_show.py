import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_inventory_show(self, uuid, resource_class, *, include_used=False):
    resource = self.openstack(f'resource provider inventory show {uuid} {resource_class}', use_json=True)
    if not include_used:
        del resource['used']
    return resource