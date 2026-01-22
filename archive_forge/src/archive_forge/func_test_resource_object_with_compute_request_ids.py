import requests
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_resource_object_with_compute_request_ids(self):
    resp_obj = create_response_obj_with_compute_header()
    r = base.Resource(None, {'name': '1'}, resp=resp_obj)
    self.assertEqual(fakes.FAKE_REQUEST_ID_LIST, r.request_ids)