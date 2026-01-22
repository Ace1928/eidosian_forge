import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_delete_security_group_rule(self, flow: str=None, from_port_range: int=None, ip_protocol: str=None, ip_range: str=None, rules: List[dict]=None, sg_account_id_to_unlink: str=None, sg_id: str=None, sg_name_to_unlink: str=None, to_port_range: int=None, dry_run: bool=False):
    """
        Deletes one or more inbound or outbound rules from a security group.
        For the rule to be deleted, the values specified in the deletion
        request must exactly match the value of the existing rule.
        In case of TCP and UDP protocols, you have to indicate the destination
        port or range of ports. In case of ICMP protocol, you have to specify
        the ICMP type and code.
        Rules (IP permissions) consist of the protocol, IP address range or
        source security group.
        To remove outbound access to a destination security group, we
        recommend to use a set of IP permissions. We also recommend to specify
        the protocol in a set of IP permissions.

        :param      flow: The direction of the flow: Inbound or Outbound.
        You can specify Outbound for Nets only.type (required)
        description: ``bool``

        :param      from_port_range: The beginning of the port range for
        the TCP and UDP protocols, or an ICMP type number.
        :type       from_port_range: ``int``

        :param      ip_range: The name The IP range for the security group
        rule, in CIDR notation (for example, 10.0.0.0/16).
        :type       ip_range: ``str``

        :param      ip_protocol: The IP protocol name (tcp, udp, icmp) or
        protocol number. By default, -1, which means all protocols.
        :type       ip_protocol: ``str``

        :param      rules: Information about the security group rule to create:
        https://docs.outscale.com/api#createsecuritygrouprule
        :type       rules: ``list`` of  ``dict``

        :param      sg_account_id_to_unlink: The account ID of the
        owner of the security group for which you want to delete a rule.
        :type       sg_account_id_to_unlink: ``str``

        :param      sg_id: The ID of the security group for which
        you want to delete a rule. (required)
        :type       sg_id: ``str``

        :param      sg_name_to_unlink: The ID of the source security
        group. If you are in the Public Cloud, you can also specify the name
        of the source security group.
        :type       sg_name_to_unlink: ``str``

        :param      to_port_range: the end of the port range for the TCP and
        UDP protocols, or an ICMP type number.
        :type       to_port_range: ``int``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Security Group Rule
        :rtype: ``dict``
        """
    action = 'DeleteSecurityGroupRule'
    data = {'DryRun': dry_run}
    if flow is not None:
        data.update({'Flow': flow})
    if ip_protocol is not None:
        data.update({'IpProtocol': ip_protocol})
    if from_port_range is not None:
        data.update({'FromPortRange': from_port_range})
    if ip_range is not None:
        data.update({'IpRange': ip_range})
    if rules is not None:
        data.update({'Rules': rules})
    if sg_name_to_unlink is not None:
        data.update({'SecurityGroupNameToUnlink': sg_name_to_unlink})
    if sg_id is not None:
        data.update({'SecurityGroupId': sg_id})
    if sg_account_id_to_unlink is not None:
        data.update({'SecurityGroupAccountIdToUnlink': sg_account_id_to_unlink})
    if to_port_range is not None:
        data.update({'ToPortRange': to_port_range})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['SecurityGroup']
    return response.json()