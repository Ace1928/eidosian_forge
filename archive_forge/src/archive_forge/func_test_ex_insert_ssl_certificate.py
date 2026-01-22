import os
import sys
import pytest
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import NTTCIS_PARAMS
from libcloud.common.nttcis import NttCisPool, NttCisVIPNode, NttCisPoolMember, NttCisAPIException
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.nttcis import NttCisLBDriver
def test_ex_insert_ssl_certificate(driver):
    net_dom_id = '6aafcf08-cb0b-432c-9c64-7371265db086 '
    cert = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/loadbalancer/fixtures/nttcis/alice.crt'
    key = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/loadbalancer/fixtures/nttcis/alice.key'
    result = driver.ex_import_ssl_domain_certificate(net_dom_id, 'alice', cert, key, description='test cert')
    assert result is True