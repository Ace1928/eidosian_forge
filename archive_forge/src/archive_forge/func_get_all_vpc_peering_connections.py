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
def get_all_vpc_peering_connections(self, vpc_peering_connection_ids=None, filters=None, dry_run=False):
    """
        Retrieve information about your VPC peering connections. You
        can filter results to return information only about those VPC
        peering connections that match your search parameters.
        Otherwise, all VPC peering connections associated with your
        account are returned.

        :type vpc_peering_connection_ids: list
        :param vpc_peering_connection_ids: A list of strings with the desired VPC
            peering connection ID's

        :type filters: list of tuples
        :param filters: A list of tuples containing filters. Each tuple
            consists of a filter key and a filter value.
            Possible filter keys are:

            * *accepter-vpc-info.cidr-block* - The CIDR block of the peer VPC.
            * *accepter-vpc-info.owner-id* - The AWS account ID of the owner 
                of the peer VPC.
            * *accepter-vpc-info.vpc-id* - The ID of the peer VPC.
            * *expiration-time* - The expiration date and time for the VPC 
                peering connection.
            * *requester-vpc-info.cidr-block* - The CIDR block of the 
                requester's VPC.
            * *requester-vpc-info.owner-id* - The AWS account ID of the 
                owner of the requester VPC.
            * *requester-vpc-info.vpc-id* - The ID of the requester VPC.
            * *status-code* - The status of the VPC peering connection.
            * *status-message* - A message that provides more information 
                about the status of the VPC peering connection, if applicable.
            
        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: list
        :return: A list of :class:`boto.vpc.vpc.VPC`
        """
    params = {}
    if vpc_peering_connection_ids:
        self.build_list_params(params, vpc_peering_connection_ids, 'VpcPeeringConnectionId')
    if filters:
        self.build_filter_params(params, dict(filters))
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_list('DescribeVpcPeeringConnections', params, [('item', VpcPeeringConnection)])