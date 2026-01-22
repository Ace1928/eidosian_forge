import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_share_transfers_5678_accept(self, **kw):
    transfer = {'transfer': self.fake_transfer}
    return (202, {}, transfer)