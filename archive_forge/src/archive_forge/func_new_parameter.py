from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def new_parameter(name, schema, value=None, validate_value=True):
    tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {name: schema}})
    schema = tmpl.param_schemata()[name]
    param = parameters.Parameter(name, schema, value)
    param.validate(validate_value)
    return param