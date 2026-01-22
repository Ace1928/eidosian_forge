from __future__ import absolute_import, division, print_function
from ipaddress import ip_interface
from ansible.module_utils._text import to_text
from ansible_collections.netbox.netbox.plugins.module_utils.netbox_utils import (

        This function should have all necessary code for endpoints within the application
        to create/update/delete the endpoint objects
        Supported endpoints:
        - aggregates
        - asns
        - fhrp_groups
        - fhrp_group_assignments
        - ipam_roles
        - ip_addresses
        - l2vpns
        - l2vpn_terminations
        - prefixes
        - rirs
        - route_targets
        - vlans
        - vlan_groups
        - vrfs
        