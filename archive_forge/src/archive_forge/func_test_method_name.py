from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_method_name(self):
    e = events.APIEventPayload(mock.ANY, 'post.end', 'POST')
    self.assertEqual('post.end', e.method_name)