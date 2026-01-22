import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_provider_aggregate_list(self, uuid):
    return self.openstack('resource provider aggregate list ' + uuid, use_json=True)