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
def generate_class_from_template(name, data, param_defaults):
    tmpl = template.Template(template_format.parse(data))
    props, attrs = TemplateResource.get_schemas(tmpl, param_defaults)
    cls = type(name, (TemplateResource,), {'properties_schema': props, 'attributes_schema': attrs, '__doc__': tmpl.t.get(tmpl.DESCRIPTION)})
    return cls