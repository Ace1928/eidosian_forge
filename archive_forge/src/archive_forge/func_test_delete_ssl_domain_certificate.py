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
def test_delete_ssl_domain_certificate(driver):
    NttCisMockHttp.type = 'LIST'
    cert_name = 'alice'
    cert = driver.ex_list_ssl_domain_certs(name=cert_name)[0]
    NttCisMockHttp.type = None
    result = driver.ex_delete_ssl_domain_certificate(cert.id)
    assert result is True