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
def test_walk_raise(self):
    sot = utils.TinyDAG()
    sot.from_dict(self.test_graph)
    bad_node = 'f'
    with testtools.ExpectedException(exceptions.SDKException):
        for node in sot.walk(timeout=1):
            if node != bad_node:
                sot.node_done(node)