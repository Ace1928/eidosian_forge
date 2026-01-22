import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_inventory_delete(self, uuid, resource_class=None):
    cmd = 'resource provider inventory delete {uuid}'.format(uuid=uuid)
    if resource_class:
        cmd += ' --resource-class ' + resource_class
    self.openstack(cmd)