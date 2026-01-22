from oslo_log import log as logging
from neutron_lib.api import validators as lib_validators
from neutron_lib.callbacks import events
from neutron_lib.callbacks import registry
from neutron_lib.services.qos import constants
def validate_rule_for_port(self, context, rule, port):
    """Return True/False for valid/invalid.

        This is only meant to be used when a rule is compatible
        with some ports/networks but not with others (depending on
        port/network properties).

        Returns True by default for backwards compatibility.
        """
    return True