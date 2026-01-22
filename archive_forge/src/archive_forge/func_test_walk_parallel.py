import concurrent.futures
import hashlib
import logging
import sys
from unittest import mock
import fixtures
import os_service_types
import testtools
import openstack
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_walk_parallel(self):
    sot = utils.TinyDAG()
    sot.from_dict(self.test_graph)
    sorted_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        for node in sot.walk(timeout=1):
            executor.submit(test_walker_fn, sot, node, sorted_list)
    self._verify_order(sot.graph, sorted_list)
    print(sorted_list)
    self.assertEqual(len(self.test_graph.keys()), len(sorted_list))