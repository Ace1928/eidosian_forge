import tempfile
import testtools
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack.tests import fakes
from openstack.tests.unit import base
def test_search_stacks_exception(self):
    self.register_uris([dict(method='GET', uri='{endpoint}/stacks'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT), status_code=404)])
    with testtools.ExpectedException(exceptions.NotFoundException):
        self.cloud.search_stacks()