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
def get_all_vpn_connections(self, vpn_connection_ids=None, filters=None, dry_run=False):
    """
        Retrieve information about your VPN_CONNECTIONs.  You can filter results to
        return information only about those VPN_CONNECTIONs that match your search
        parameters.  Otherwise, all VPN_CONNECTIONs associated with your account
        are returned.

        :type vpn_connection_ids: list
        :param vpn_connection_ids: A list of strings with the desired VPN_CONNECTION ID's

        :type filters: list of tuples or dict
        :param filters: A list of tuples or dict containing filters.  Each tuple
                        or dict item consists of a filter key and a filter value.
                        Possible filter keys are:

                        - *state*, a list of states of the VPN_CONNECTION
                          pending,available,deleting,deleted
                        - *type*, a list of types of connection, currently 'ipsec.1'
                        - *customerGatewayId*, a list of IDs of the customer gateway
                          associated with the VPN
                        - *vpnGatewayId*, a list of IDs of the VPN gateway associated
                          with the VPN connection

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: list
        :return: A list of :class:`boto.vpn_connection.vpnconnection.VpnConnection`
        """
    params = {}
    if vpn_connection_ids:
        self.build_list_params(params, vpn_connection_ids, 'VpnConnectionId')
    if filters:
        self.build_filter_params(params, filters)
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_list('DescribeVpnConnections', params, [('item', VpnConnection)])