import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_allocation_delete(self, consumer_uuid):
    cmd = 'resource provider allocation delete ' + consumer_uuid
    return self.openstack(cmd)