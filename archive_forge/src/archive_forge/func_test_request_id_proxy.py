import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_request_id_proxy(self):

    def test_data(val):
        resp = create_response_obj_with_req_id(REQUEST_ID)
        return (val, resp)
    proxy = utils.RequestIdProxy(test_data(11))
    self.assertEqual([REQUEST_ID], proxy.request_ids)