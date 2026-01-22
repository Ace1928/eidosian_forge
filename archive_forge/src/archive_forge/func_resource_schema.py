import contextlib
from oslo_log import log as logging
from urllib import parse
from webob import exc
from heat.api.openstack.v1 import util
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import environment_format
from heat.common.i18n import _
from heat.common import identifier
from heat.common import param_utils
from heat.common import serializers
from heat.common import template_format
from heat.common import urlfetch
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
@util.registered_policy_enforce
def resource_schema(self, req, type_name, with_description=False):
    """Returns the schema of the given resource type."""
    return self.rpc_client.resource_schema(req.context, type_name, self._extract_bool_param('with_description', with_description))