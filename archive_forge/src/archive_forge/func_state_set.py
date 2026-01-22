import collections
import contextlib
import datetime as dt
import itertools
import pydoc
import re
import tenacity
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import short_id
from heat.common import timeutils
from heat.engine import attributes
from heat.engine.cfn import template as cfn_tmpl
from heat.engine import clients
from heat.engine.clients import default_client_plugin
from heat.engine import environment
from heat.engine import event
from heat.engine import function
from heat.engine.hot import template as hot_tmpl
from heat.engine import node_data
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import status
from heat.engine import support
from heat.engine import sync_point
from heat.engine import template
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_objects
from heat.objects import resource_properties_data as rpd_objects
from heat.rpc import client as rpc_client
def state_set(self, action, status, reason='state changed', lock=LOCK_NONE):
    if action not in self.ACTIONS:
        raise ValueError(_('Invalid action %s') % action)
    if status not in self.STATUSES:
        raise ValueError(_('Invalid status %s') % status)
    old_state = (self.action, self.status)
    new_state = (action, status)
    set_metadata = self.action == self.INIT
    self.action = action
    self.status = status
    self.status_reason = reason
    self.store(set_metadata, lock=lock)
    if new_state != old_state:
        self._add_event(action, status, reason)
    if status != self.COMPLETE:
        self.clear_stored_attributes()