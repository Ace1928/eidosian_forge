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
def create_route(self, route_table_id, destination_cidr_block, gateway_id=None, instance_id=None, interface_id=None, vpc_peering_connection_id=None, dry_run=False):
    """
        Creates a new route in the route table within a VPC. The route's target
        can be either a gateway attached to the VPC or a NAT instance in the
        VPC.

        :type route_table_id: str
        :param route_table_id: The ID of the route table for the route.

        :type destination_cidr_block: str
        :param destination_cidr_block: The CIDR address block used for the
                                       destination match.

        :type gateway_id: str
        :param gateway_id: The ID of the gateway attached to your VPC.

        :type instance_id: str
        :param instance_id: The ID of a NAT instance in your VPC.

        :type interface_id: str
        :param interface_id: Allows routing to network interface attachments.

        :type vpc_peering_connection_id: str
        :param vpc_peering_connection_id: Allows routing to VPC peering
                                          connection.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    params = {'RouteTableId': route_table_id, 'DestinationCidrBlock': destination_cidr_block}
    if gateway_id is not None:
        params['GatewayId'] = gateway_id
    elif instance_id is not None:
        params['InstanceId'] = instance_id
    elif interface_id is not None:
        params['NetworkInterfaceId'] = interface_id
    elif vpc_peering_connection_id is not None:
        params['VpcPeeringConnectionId'] = vpc_peering_connection_id
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('CreateRoute', params)