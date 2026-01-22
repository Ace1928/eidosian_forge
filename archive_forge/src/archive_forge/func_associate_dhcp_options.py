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
def associate_dhcp_options(self, dhcp_options_id, vpc_id, dry_run=False):
    """
        Associate a set of Dhcp Options with a VPC.

        :type dhcp_options_id: str
        :param dhcp_options_id: The ID of the Dhcp Options

        :type vpc_id: str
        :param vpc_id: The ID of the VPC.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    params = {'DhcpOptionsId': dhcp_options_id, 'VpcId': vpc_id}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('AssociateDhcpOptions', params)