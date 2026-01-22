import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_provider_show_usage(self, uuid):
    return self.openstack('resource provider usage show ' + uuid, use_json=True)