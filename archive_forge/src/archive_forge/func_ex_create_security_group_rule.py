import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_security_group_rule(self, flow: str=None, from_port_range: int=None, ip_range: str=None, rules: List[dict]=None, sg_account_id_to_link: str=None, sg_id: str=None, sg_name_to_link: str=None, to_port_range: int=None, dry_run: bool=False):
    """
        Configures the rules for a security group.
        The modifications are effective at virtual machine (VM) level as
        quickly as possible, but a small delay may occur.

        You can add one or more egress rules to a security group for use with
        a Net.
        It allows VMs to send traffic to either one or more destination IP
        address ranges or destination security groups for the same Net.
        We recommend using a set of IP permissions to authorize outbound
        access to a destination security group. We also recommended this
        method to create a rule with a specific IP protocol and a specific
        port range. In a set of IP permissions, we recommend to specify the
        the protocol.

        You can also add one or more ingress rules to a security group.
        In the public Cloud, this action allows one or more IP address ranges
        to access a security group for your account, or allows one or more
        security groups (source groups) to access a security group for your
        own 3DS OUTSCALE account or another one.
        In a Net, this action allows one or more IP address ranges to access a
        security group for your Net, or allows one or more other security
        groups (source groups) to access a security group for your Net. All
        the security groups must be for the same Net.

        :param      flow: The direction of the flow: Inbound or Outbound.
        You can specify Outbound for Nets only.type (required)
        description: ``bool``

        :param      from_port_range: The beginning of the port range for
        the TCP and UDP protocols, or an ICMP type number.
        :type       from_port_range: ``int``

        :param      ip_range: The name The IP range for the security group
        rule, in CIDR notation (for example, 10.0.0.0/16).
        :type       ip_range: ``str``

        :param      rules: Information about the security group rule to create:
        https://docs.outscale.com/api#createsecuritygrouprule
        :type       rules: ``list`` of  ``dict``

        :param      sg_account_id_to_link: The account ID of the
        owner of the security group for which you want to create a rule.
        :type       sg_account_id_to_link: ``str``

        :param      sg_id: The ID of the security group for which
        you want to create a rule. (required)
        :type       sg_id: ``str``

        :param      sg_name_to_link: The ID of the source security
        group. If you are in the Public Cloud, you can also specify the name
        of the source security group.
        :type       sg_name_to_link: ``str``

        :param      to_port_range: the end of the port range for the TCP and
        UDP protocols, or an ICMP type number.
        :type       to_port_range: ``int``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: The new Security Group Rule
        :rtype: ``dict``
        """
    action = 'CreateSecurityGroupRule'
    data = {'DryRun': dry_run}
    if flow is not None:
        data.update({'Flow': flow})
    if from_port_range is not None:
        data.update({'FromPortRange': from_port_range})
    if ip_range is not None:
        data.update({'IpRange': ip_range})
    if rules is not None:
        data.update({'Rules': rules})
    if sg_name_to_link is not None:
        data.update({'SecurityGroupNameToLink': sg_name_to_link})
    if sg_id is not None:
        data.update({'SecurityGroupId': sg_id})
    if sg_account_id_to_link is not None:
        data.update({'SecurityGroupAccountIdToLink': sg_account_id_to_link})
    if to_port_range is not None:
        data.update({'ToPortRange': to_port_range})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['SecurityGroup']
    return response.json()