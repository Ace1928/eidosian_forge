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
def generate_template(self, req, type_name):
    """Generates a template based on the specified type."""
    template_type = 'cfn'
    if rpc_api.TEMPLATE_TYPE in req.params:
        try:
            template_type = param_utils.extract_template_type(req.params.get(rpc_api.TEMPLATE_TYPE))
        except ValueError as ex:
            msg = _('Template type is not supported: %s') % ex
            raise exc.HTTPBadRequest(str(msg))
    return self.rpc_client.generate_template(req.context, type_name, template_type)