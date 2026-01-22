from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
@staticmethod
def prepare_properties(properties, name):
    """Prepares the property values for correct Neutron create call.

        Prepares the property values so that they can be passed directly to
        the Neutron create call.

        Removes None values and value_specs, merges value_specs with the main
        values.
        """
    props = dict(((k, v) for k, v in properties.items() if v is not None))
    if 'name' in properties:
        props.setdefault('name', name)
    if 'value_specs' in props:
        NeutronResource.merge_value_specs(props)
    return props