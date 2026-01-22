import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from swiftclient import client as swiftclient_client
from swiftclient import exceptions as swiftclient_exceptions
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import swift
from heat.engine import node_data
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template as templatem
from heat.tests import common
from heat.tests import utils
def test_swift_handle_refid_convergence_cache_data(self):
    cache_data = {'test_wait_condition_handle': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'})}
    st = create_stack(swiftsignalhandle_template, cache_data=cache_data)
    rsrc = st.defn['test_wait_condition_handle']
    self.assertEqual('convg_xyz', rsrc.FnGetRefId())