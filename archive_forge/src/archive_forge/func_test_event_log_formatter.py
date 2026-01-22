import os
from unittest import mock
from urllib import request
import testtools
from heatclient.common import utils
from heatclient import exc
from heatclient.v1 import resources as hc_res
def test_event_log_formatter(self):
    event1 = {'event_time': '2015-09-28T12:12:12', 'id': '123456789', 'resource_name': 'res_name', 'resource_status': 'CREATE_IN_PROGRESS', 'resource_status_reason': 'CREATE started'}
    event2 = {'event_time': '2015-09-28T12:12:22', 'id': '123456789', 'resource_name': 'res_name', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'CREATE completed'}
    events_list = [hc_res.Resource(manager=None, info=event1), hc_res.Resource(manager=None, info=event2)]
    expected = '2015-09-28 12:12:12 [res_name]: CREATE_IN_PROGRESS  CREATE started\n2015-09-28 12:12:22 [res_name]: CREATE_COMPLETE  CREATE completed'
    self.assertEqual(expected, utils.event_log_formatter(events_list))
    self.assertEqual('', utils.event_log_formatter([]))