import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_get_href(self):
    xml = '\n<datacenter>\n        <link href="http://10.60.12.7:80/api/admin/datacenters/2"\n        type="application/vnd.abiquo.datacenter+xml" rel="edit1"/>\n        <link href="http://10.60.12.7:80/ponies/bar/foo/api/admin/datacenters/3"\n        type="application/vnd.abiquo.datacenter+xml" rel="edit2"/>\n        <link href="http://vdcbridge.interoute.com:80/jclouds/apiouds/api/admin/enterprises/1234"\n        type="application/vnd.abiquo.datacenter+xml" rel="edit3"/>\n</datacenter>\n'
    elem = ET.XML(xml)
    href = get_href(element=elem, rel='edit1')
    self.assertEqual(href, '/admin/datacenters/2')
    href = get_href(element=elem, rel='edit2')
    self.assertEqual(href, '/admin/datacenters/3')
    href = get_href(element=elem, rel='edit3')
    self.assertEqual(href, '/admin/enterprises/1234')