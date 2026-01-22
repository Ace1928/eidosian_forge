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
def create_subnet(self, vpc_id, cidr_block, availability_zone=None, dry_run=False):
    """
        Create a new Subnet

        :type vpc_id: str
        :param vpc_id: The ID of the VPC where you want to create the subnet.

        :type cidr_block: str
        :param cidr_block: The CIDR block you want the subnet to cover.

        :type availability_zone: str
        :param availability_zone: The AZ you want the subnet in

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: The newly created Subnet
        :return: A :class:`boto.vpc.customergateway.Subnet` object
        """
    params = {'VpcId': vpc_id, 'CidrBlock': cidr_block}
    if availability_zone:
        params['AvailabilityZone'] = availability_zone
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('CreateSubnet', params, Subnet)