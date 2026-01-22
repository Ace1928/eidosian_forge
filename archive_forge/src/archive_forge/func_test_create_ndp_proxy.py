from openstackclient.tests.functional.network.v2 import common
def test_create_ndp_proxy(self):
    ndp_proxies = [{'name': self.getUniqueString(), 'router_id': self.ROT_ID, 'port_id': self.INT_PORT_ID, 'address': self.INT_PORT_ADDRESS}]
    self._create_ndp_proxies(ndp_proxies)