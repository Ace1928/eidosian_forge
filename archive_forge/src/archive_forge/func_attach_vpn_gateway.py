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
def attach_vpn_gateway(self, vpn_gateway_id, vpc_id, dry_run=False):
    """
        Attaches a VPN gateway to a VPC.

        :type vpn_gateway_id: str
        :param vpn_gateway_id: The ID of the vpn_gateway to attach

        :type vpc_id: str
        :param vpc_id: The ID of the VPC you want to attach the gateway to.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: An attachment
        :return: a :class:`boto.vpc.vpngateway.Attachment`
        """
    params = {'VpnGatewayId': vpn_gateway_id, 'VpcId': vpc_id}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('AttachVpnGateway', params, Attachment)