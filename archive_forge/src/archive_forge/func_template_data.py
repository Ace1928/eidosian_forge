from oslo_log import log as logging
from oslo_serialization import jsonutils
from requests import exceptions
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import template
from heat.rpc import api as rpc_api
def template_data(self):
    reported_excp = None
    t_data = self.stack.t.files.get(self.template_url)
    stored_t_data = t_data
    if t_data is None:
        LOG.debug('TemplateResource data file "%s" not found in files.', self.template_url)
    if not t_data and self.template_url.endswith(('.yaml', '.template')):
        try:
            t_data = self.get_template_file(self.template_url, self.allowed_schemes)
        except exception.NotFound as err:
            if self.action == self.UPDATE:
                raise
            reported_excp = err
    if t_data is None:
        nested_identifier = self.nested_identifier()
        if nested_identifier is not None:
            nested_t = self.rpc_client().get_template(self.context, nested_identifier)
            t_data = jsonutils.dumps(nested_t)
    if t_data is not None:
        if t_data != stored_t_data:
            self.stack.t.files[self.template_url] = t_data
        self.stack.t.env.register_class(self.resource_type, self.template_url, path=self.resource_path)
        return t_data
    if reported_excp is None:
        reported_excp = ValueError(_('Unknown error retrieving %s') % self.template_url)
    raise reported_excp