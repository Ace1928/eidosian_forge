from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import net
from heat.engine import support
@staticmethod
def prepare_provider_properties(props):
    if ProviderNet.PROVIDER_NETWORK_TYPE in props:
        ProviderNet.add_provider_extension(props, ProviderNet.PROVIDER_NETWORK_TYPE)
    if ProviderNet.PROVIDER_PHYSICAL_NETWORK in props:
        ProviderNet.add_provider_extension(props, ProviderNet.PROVIDER_PHYSICAL_NETWORK)
    if ProviderNet.PROVIDER_SEGMENTATION_ID in props:
        ProviderNet.add_provider_extension(props, ProviderNet.PROVIDER_SEGMENTATION_ID)
    if ProviderNet.ROUTER_EXTERNAL in props:
        props['router:external'] = props.pop(ProviderNet.ROUTER_EXTERNAL)