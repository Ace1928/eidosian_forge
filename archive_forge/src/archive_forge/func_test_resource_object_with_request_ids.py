from unittest import mock
import requests
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient.tests.unit import test_utils
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import client
from cinderclient.v3 import volumes
def test_resource_object_with_request_ids(self):
    resp_obj = create_response_obj_with_header()
    r = base.Resource(None, {'name': '1'}, resp=resp_obj)
    self.assertEqual([REQUEST_ID], r.request_ids)