from oslo_log import log as logging
from neutron_lib.api import validators as lib_validators
from neutron_lib.callbacks import events
from neutron_lib.callbacks import registry
from neutron_lib.services.qos import constants
def is_vif_type_compatible(self, vif_type):
    """True if the driver is compatible with the VIF type."""
    return vif_type in self.vif_types