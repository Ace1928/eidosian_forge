import copy
from unittest import mock
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.allocation
def test_allocations_list_sort_dir(self):
    allocations = self.mgr.list(sort_dir='desc')
    expect = [('GET', '/v1/allocations/?sort_dir=desc', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(allocations))
    expected_resp = ({}, {'allocations': [ALLOCATION2, ALLOCATION]})
    self.assertEqual(expected_resp, self.api.responses['/v1/allocations/?sort_dir=desc']['GET'])