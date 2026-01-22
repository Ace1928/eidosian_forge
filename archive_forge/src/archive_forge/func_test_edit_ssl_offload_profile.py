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
def test_edit_ssl_offload_profile(driver):
    profile_name = 'ssl_offload'
    datacenter_id = 'EU6'
    NttCisMockHttp.type = 'LIST'
    profile = driver.ex_list_ssl_offload_profiles(name=profile_name, datacenter_id=datacenter_id)[0]
    NttCisMockHttp.type = None
    result = driver.ex_edit_ssl_offload_profile(profile.id, profile.name, profile.sslDomainCertificate.id, ciphers=profile.ciphers, description='A test edit of an offload profile')
    assert result is True