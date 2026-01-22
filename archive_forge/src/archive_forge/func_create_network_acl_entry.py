from boto.ec2.connection import EC2Connection
from boto.resultset import ResultSet
from boto.vpc.vpc import VPC
from boto.vpc.customergateway import CustomerGateway
from boto.vpc.networkacl import NetworkAcl
from boto.vpc.routetable import RouteTable
from boto.vpc.internetgateway import InternetGateway
from boto.vpc.vpngateway import VpnGateway, Attachment
from boto.vpc.dhcpoptions import DhcpOptions
from boto.vpc.subnet import Subnet
from boto.vpc.vpnconnection import VpnConnection
from boto.vpc.vpc_peering_connection import VpcPeeringConnection
from boto.ec2 import RegionData
from boto.regioninfo import RegionInfo, get_regions
from boto.regioninfo import connect
def create_network_acl_entry(self, network_acl_id, rule_number, protocol, rule_action, cidr_block, egress=None, icmp_code=None, icmp_type=None, port_range_from=None, port_range_to=None):
    """
        Creates a new network ACL entry in a network ACL within a VPC.

        :type network_acl_id: str
        :param network_acl_id: The ID of the network ACL for this network ACL entry.

        :type rule_number: int
        :param rule_number: The rule number to assign to the entry (for example, 100).

        :type protocol: int
        :param protocol: Valid values: -1 or a protocol number
        (http://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml)

        :type rule_action: str
        :param rule_action: Indicates whether to allow or deny traffic that matches the rule.

        :type cidr_block: str
        :param cidr_block: The CIDR range to allow or deny, in CIDR notation (for example,
        172.16.0.0/24).

        :type egress: bool
        :param egress: Indicates whether this rule applies to egress traffic from the subnet (true)
        or ingress traffic to the subnet (false).

        :type icmp_type: int
        :param icmp_type: For the ICMP protocol, the ICMP type. You can use -1 to specify
         all ICMP types.

        :type icmp_code: int
        :param icmp_code: For the ICMP protocol, the ICMP code. You can use -1 to specify
        all ICMP codes for the given ICMP type.

        :type port_range_from: int
        :param port_range_from: The first port in the range.

        :type port_range_to: int
        :param port_range_to: The last port in the range.


        :rtype: bool
        :return: True if successful
        """
    params = {'NetworkAclId': network_acl_id, 'RuleNumber': rule_number, 'Protocol': protocol, 'RuleAction': rule_action, 'CidrBlock': cidr_block}
    if egress is not None:
        if isinstance(egress, bool):
            egress = str(egress).lower()
        params['Egress'] = egress
    if icmp_code is not None:
        params['Icmp.Code'] = icmp_code
    if icmp_type is not None:
        params['Icmp.Type'] = icmp_type
    if port_range_from is not None:
        params['PortRange.From'] = port_range_from
    if port_range_to is not None:
        params['PortRange.To'] = port_range_to
    return self.get_status('CreateNetworkAclEntry', params)