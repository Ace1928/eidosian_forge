from breezy import tests
from breezy.bzr import btree_index
from breezy.tests import http_server
def test_dump_btree_http_smoke(self):
    self.transport_readonly_server = http_server.HttpServer
    self.create_sample_btree_index()
    url = self.get_readonly_url('test.btree')
    out, err = self.run_bzr(['dump-btree', url])
    self.assertEqualDiff("(('test', 'key1'), 'value', ((('ref', 'entry'),),))\n(('test', 'key2'), 'value2', ((('ref', 'entry2'),),))\n(('test2', 'key3'), 'value3', ((('ref', 'entry3'),),))\n", out)