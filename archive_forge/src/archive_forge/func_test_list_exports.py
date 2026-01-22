import time
import uuid
from designateclient.tests import v2
def test_list_exports(self):
    items = [self.new_ref(), self.new_ref()]
    parts = ['zones', 'tasks', 'exports']
    self.stub_url('GET', parts=parts, json={'exports': items})
    listed = self.client.zone_exports.list()
    self.assertList(items, listed['exports'])
    self.assertQueryStringIs('')