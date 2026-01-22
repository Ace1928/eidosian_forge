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
def create_internet_gateway(self, dry_run=False):
    """
        Creates an internet gateway for VPC.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: Newly created internet gateway.
        :return: `boto.vpc.internetgateway.InternetGateway`
        """
    params = {}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('CreateInternetGateway', params, InternetGateway)