import copy
import uuid
import netaddr
from oslo_config import cfg
from oslo_utils import strutils
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.placement import utils as pl_utils
from neutron_lib.utils import net as net_utils
def convert_to_sanitized_binding_profile_allocation(allocation, port_id, min_bw_rules):
    """Return binding-profile.allocation in the new format

    :param allocation: binding-profile.allocation attribute containting a
                       string with RP UUID
    :param port_id: ID of the port that is being sanitized
    :param min_bw_rules: A list of minimum bandwidth rules associated with the
                         port.
    :return: A dict with allocation in {'<group_uuid>': '<rp_uuid>'} format.
    """
    if isinstance(allocation, dict):
        return allocation
    group_id = str(pl_utils.resource_request_group_uuid(uuid.UUID(port_id), min_bw_rules))
    return {group_id: allocation}