from oslo_log import log as logging
from neutron_lib.api import validators as lib_validators
from neutron_lib.callbacks import events
from neutron_lib.callbacks import registry
from neutron_lib.services.qos import constants
def validate_rule_for_network(self, context, rule, network_id):
    """Return True/False for valid/invalid.

        This is only meant to be used when a rule is compatible
        with some networks but not with others (depending on
        network properties).

        Returns True by default for backwards compatibility.
        """
    return True