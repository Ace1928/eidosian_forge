import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
def test_format_event_legacy_prop_data(self):
    event = self._dummy_event(res_properties=None)
    with db_api.context_manager.writer.using(self.stack.context):
        db_obj = self.stack.context.session.query(models.Event).filter_by(id=event.id).first()
        db_obj.update({'resource_properties': {'legacy_k1': 'legacy_v1'}})
        db_obj.save(self.stack.context.session)
    event_legacy = event_object.Event.get_all_by_stack(self.context, self.stack.id)[0]
    formatted = api.format_event(event_legacy, self.stack.identifier())
    self.assertEqual({'legacy_k1': 'legacy_v1'}, formatted[rpc_api.EVENT_RES_PROPERTIES])