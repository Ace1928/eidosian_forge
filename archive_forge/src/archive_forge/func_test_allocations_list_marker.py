import copy
from unittest import mock
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.allocation
def test_allocations_list_marker(self):
    allocations = self.mgr.list(marker=ALLOCATION['uuid'])
    expect = [('GET', '/v1/allocations/?marker=%s' % ALLOCATION['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(allocations))
    expected_resp = ({}, {'next': 'http://127.0.0.1:6385/v1/allocations/?limit=1', 'allocations': [ALLOCATION]})
    self.assertEqual(expected_resp, self.api.responses['/v1/allocations']['GET'])