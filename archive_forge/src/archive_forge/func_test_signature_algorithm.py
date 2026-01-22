import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
def test_signature_algorithm(self):
    cases = [({'command': 'listVirtualMachines'}, 'z/a9Y7J52u48VpqIgiwaGUMCso0='), ({'command': 'deployVirtualMachine', 'name': 'fred', 'displayname': 'George', 'serviceofferingid': 5, 'templateid': 17, 'zoneid': 23, 'networkids': 42}, 'gHTo7mYmadZ+zluKHzlEKb1i/QU='), ({'command': 'deployVirtualMachine', 'name': 'fred', 'displayname': 'George+Ringo', 'serviceofferingid': 5, 'templateid': 17, 'zoneid': 23, 'networkids': 42}, 'tAgfrreI1ZvWlWLClD3gu4+aKv4=')]
    connection = CloudStackConnection('fnord', 'abracadabra')
    for case in cases:
        params = connection.add_default_params(case[0])
        self.assertEqual(connection._make_signature(params), b(case[1]))