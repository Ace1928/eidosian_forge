from unittest import mock
import ddt
from os_brick.initiator.connectors import base_rbd
from os_brick.tests import base
@ddt.data((['192.168.1.1', '192.168.1.2'], ['192.168.1.1', '192.168.1.2']), (['3ffe:1900:4545:3:200:f8ff:fe21:67cf', 'fe80:0:0:0:200:f8ff:fe21:67cf'], ['[3ffe:1900:4545:3:200:f8ff:fe21:67cf]', '[fe80:0:0:0:200:f8ff:fe21:67cf]']), (['foobar', 'fizzbuzz'], ['foobar', 'fizzbuzz']), (['192.168.1.1', '3ffe:1900:4545:3:200:f8ff:fe21:67cf', 'hello, world!'], ['192.168.1.1', '[3ffe:1900:4545:3:200:f8ff:fe21:67cf]', 'hello, world!']))
@ddt.unpack
def test_sanitize_mon_host(self, hosts_in, hosts_out):
    self.assertEqual(hosts_out, self._conn._sanitize_mon_hosts(hosts_in))