from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
@staticmethod
def merge_value_specs(props, before_value_specs=None):
    value_spec_props = props.pop('value_specs')
    if value_spec_props is not None:
        if before_value_specs:
            for k in list(value_spec_props):
                if value_spec_props[k] == before_value_specs.get(k, None):
                    value_spec_props.pop(k)
        props.update(value_spec_props)