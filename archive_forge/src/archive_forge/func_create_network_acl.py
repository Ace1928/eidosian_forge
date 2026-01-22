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
def create_network_acl(self, vpc_id):
    """
        Creates a new network ACL.

        :type vpc_id: str
        :param vpc_id: The VPC ID to associate this network ACL with.

        :rtype: The newly created network ACL
        :return: A :class:`boto.vpc.networkacl.NetworkAcl` object
        """
    params = {'VpcId': vpc_id}
    return self.get_object('CreateNetworkAcl', params, NetworkAcl)